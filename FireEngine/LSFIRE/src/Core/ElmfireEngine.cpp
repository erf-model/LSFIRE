#include "ElmfireEngine.H"
#include "../Physics/Rothermel.H"
#include "../Physics/EllipticalFire.H"
#include "../Numerics/Gradients.H"
#include "../Numerics/LevelSetSolver.H"
#include <AMReX_MFIter.H>
#include <AMReX_Gpu.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <cmath>
#include <cstdio>

using namespace amrex;

// ============================================================================
// FILE-SCOPE KERNEL LAUNCHERS
// nvcc forbids __device__ lambdas in private/protected member functions;
// these static helpers host the ParallelFor so the class methods stay private.
// ============================================================================

static void LaunchSurfaceSpreadKernel(FireNodeData* nodes_d,
                                       FuelModelTable* tables_dp,
                                       int N)
{
    amrex::ParallelFor(N, [nodes_d, tables_dp] AMREX_GPU_DEVICE (int i) noexcept {
        FireNodeData& nd = nodes_d[i];
        if (nd.ifbfm == 0) {
            nd.vs0 = 0.0;
            nd.velocity_dms_surface = 0.0;
            nd.ux = 0.0;
            nd.uy = 0.0;
            nd.vback = 0.0;
            return;
        }
        SurfaceSpreadRate(nd, tables_dp[i]);
    });
}

static void LaunchEllipticalFireKernel(FireNodeData* nodes_d,
                                        FuelModelTable* tables_dp,
                                        int N)
{
    amrex::ParallelFor(N, [nodes_d, tables_dp] AMREX_GPU_DEVICE (int i) noexcept {
        FireNodeData& nd = nodes_d[i];
        if (nd.ifbfm == 0) {
            nd.ux = 0.0;
            nd.uy = 0.0;
            return;
        }
        EllipticalFireVelocity(nd, tables_dp[i], amrex::Real(1.0));
    });
}

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

ElmfireEngine::ElmfireEngine()
    : m_time(0.0)
    , m_fuel_db(nullptr)
    , m_uniform_fuel_id(1)
    , m_band_thickness(5)
    , m_wind_speed(0.0)
    , m_wind_direction(270.0)
    , m_m1(0.06)
    , m_m10(0.08)
    , m_m100(0.10)
    , m_mlh(0.50)
    , m_mlw(1.00)
    , m_slope_x(0.0)
    , m_slope_y(0.0)
    , m_tanslp2(0.0)
    , m_wind_data_scale_factor(0.7)
    , m_target_cfl(0.9)
    , m_cellsize(30.0)
    , m_rcellsize(1.0/30.0)
    , m_use_grid_aware_burnout(false)
    , m_use_source_fuel_scaling(false)
    , m_source_fuel_exponent(1.0)
    , m_terrain_scale_factor(1.0)
    , m_has_terrain_slope(false)
{
}

ElmfireEngine::~ElmfireEngine()
{
}

// ============================================================================
// INITIALIZATION
// ============================================================================

void ElmfireEngine::Initialize(const Geometry& geom,
                                FuelModelDatabase& fuel_database,
                                const BoxArray* ba_in,
                                const DistributionMapping* dm_in)
{
    m_geom = geom;
    m_fuel_db = &fuel_database;

    const Real* dx = geom.CellSize();
    m_cellsize = dx[0];
    m_rcellsize = 1.0 / m_cellsize;

    BoxArray ba;
    DistributionMapping dm;

    if (ba_in != nullptr && dm_in != nullptr) {
        ba = *ba_in;
        dm = *dm_in;
        Print() << "ElmfireEngine: Using ERF's BoxArray decomposition (" << ba.size() << " boxes)\n";
    } else {
        ba.define(geom.Domain());
        if (ParallelDescriptor::NProcs() > 1) {
            int nprocs = ParallelDescriptor::NProcs();
            const Box& domain = geom.Domain();
            int nx = domain.length(0);
            int ny = domain.length(1);
            int target_boxes = 2 * nprocs;
            int max_size = std::max(32, static_cast<int>(std::sqrt(
                static_cast<double>(nx) * ny / target_boxes)));
            ba.maxSize(max_size);
        }
        dm.define(ba);
        Print() << "ElmfireEngine: Creating independent BoxArray decomposition (" << ba.size() << " boxes)\n";
    }

    // Solver stencils need three cells; cross-rank narrow-band activation
    // needs the full physical band width around a newly burned cell.
    int ngrow = std::max(3, m_band_thickness);
    // Use managed (unified) memory so CPU narrow-band loops can read phi without
    // explicit host copies.  On CPU builds The_Managed_Arena()==The_Arena().
    MFInfo phi_info;
    phi_info.SetArena(The_Managed_Arena());
    m_phi.define(ba, dm, 1, ngrow, phi_info);
    m_phi_correction.define(ba, dm, 1, 0, phi_info);
    m_fire_history.define(ba, dm, NumFireHistoryComp, 0, phi_info);
    m_fire_history.setVal(0.0);
    m_fire_history.setVal(1.0, FuelFrac, 1);
    m_fire_history.setVal(-1.0, IgnitionTime, 1);
    m_fire_history.setVal(600.0, BurnoutTime, 2);
    m_fire_history.setVal(1.0, FluxScaling, 1);
    m_fire_history.setVal(-1.0, SpotEmitter, 1);   // not decided
    m_fuel_model.define(ba, dm, 1, ngrow, phi_info);

    m_phi.setVal(1.0);
    m_fuel_model.setVal(1.0);

    m_spot.Read();
    m_spot.Print();
    m_ember_count.define(ba, dm, 1, 0, phi_info);
    m_spot_ignited.define(ba, dm, 1, 0, phi_info);
    m_ember_count.setVal(0.0);
    m_spot_ignited.setVal(0.0);

    {
        ParmParse pe("fire.extinction");
        pe.query("enable", m_extinction);
        if (m_extinction) {
            m_dead.define(ba, dm, 1, ngrow, phi_info);
            m_dead.setVal(0.0);
            pe.queryarr("test_box", m_ext_test_box);
            pe.query("test_time", m_ext_test_time);
            if (!m_ext_test_box.empty() && m_ext_test_box.size() != 4)
                Abort("fire.extinction.test_box needs 4 values: x0 y0 x1 y1 [m]");
            Print() << "Extinction (dead burned cells masked as +1, no phi rise next to them) enabled";
            if (m_ext_test_box.size() == 4)
                Print() << "; test: cells burned by t = " << m_ext_test_time << " s in [" << m_ext_test_box[0] << ", "
                        << m_ext_test_box[2] << "] x [" << m_ext_test_box[1] << ", " << m_ext_test_box[3] << "] m die";
            Print() << "\n";
        }
    }

    m_time = 0.0;

    MFIter::allowMultipleMFIters(true);

    Print() << "ElmfireEngine initialized:\n";
    Print() << "  Domain: " << geom.Domain() << "\n";
    Print() << "  Cell size: " << m_cellsize << " m\n";
    Print() << "  CFL: " << m_target_cfl << "\n";
}

void ElmfireEngine::SetIgnition(Real x, Real y, Real radius, Real time)
{
    (void)radius;
    (void)time;

    const Real* prob_lo = m_geom.ProbLo();
    const Real* dx = m_geom.CellSize();

    int ix_ign = static_cast<int>((x - prob_lo[0]) / dx[0]);
    int iy_ign = static_cast<int>((y - prob_lo[1]) / dx[1]);

    if (ParallelDescriptor::IOProcessor()) {
        Print() << "\n========================================\n";
        Print() << "IGNITION LOCATION CALCULATION\n";
        Print() << "========================================\n";
        Print() << "Input ignition coordinates:\n";
        Print() << "  x = " << x << " m\n";
        Print() << "  y = " << y << " m\n";
        Print() << "Geometry information:\n";
        Print() << "  prob_lo = (" << prob_lo[0] << ", " << prob_lo[1] << ", " << prob_lo[2] << ") m\n";
        Print() << "  dx = (" << dx[0] << ", " << dx[1] << ", " << dx[2] << ") m\n";
        Print() << "Calculated grid indices:\n";
        Print() << "  ix_ign = (" << x << " - " << prob_lo[0] << ") / " << dx[0] << " = " << ix_ign << "\n";
        Print() << "  iy_ign = (" << y << " - " << prob_lo[1] << ") / " << dx[1] << " = " << iy_ign << "\n";
        Print() << "POINT IGNITION at cell (" << ix_ign << ", " << iy_ign << ")\n";
        Print() << "  Physical location: (" << x << ", " << y << ") m\n";
        Print() << "  (Fortran-style: phi=1 everywhere, phi=-1 at ignition cell)\n";
        Print() << "========================================\n\n";
    }

    for (MFIter mfi(m_phi); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto& phi_arr = m_phi.array(mfi);

        if (box.contains(IntVect(amrex::Dim3{ix_ign, iy_ign, 0}))) {
            phi_arr(ix_ign, iy_ign, 0) = -1.0;
            Print() << "  [Rank " << ParallelDescriptor::MyProc()
                   << "] Set phi(ignition cell) = -1.0, box = " << box << "\n";
        }
    }

    m_phi.FillBoundary(m_geom.periodicity());
    TagNarrowBand();

    for (int i = 0; i < m_narrow_band.GetNumNodes(); i++) {
        FireNodeData& nd = m_narrow_band[i];
        if (nd.burned && nd.ix == ix_ign && nd.iy == iy_ign) {
            nd.fuel_frac        = 1.0;
            nd.time_of_ignition = m_time;

            if (m_fuel_db && m_fuel_db->HasFuelModel(nd.ifbfm)) {
                FuelModelTable table = m_fuel_db->GetFuelModelTable(nd.ifbfm, nd.mlh);
                nd.burnout_time_table = table.burnout_time;
                const Real R_min = 1.0e-3;
                Real tau_transit  = m_cellsize / R_min;
                nd.burnout_time = table.burnout_time + tau_transit;
                nd.flux_scaling = table.burnout_time / nd.burnout_time;
            }
            break;
        }
    }
    SaveNodeHistory();
}

void ElmfireEngine::SetAreaIgnition(const amrex::iMultiFab* ignition_mask)
{
    using namespace amrex;

    if (ignition_mask == nullptr) {
        Abort("ElmfireEngine::SetAreaIgnition: ignition_mask is nullptr");
    }

    int num_ignited = 0;

    for (MFIter mfi(m_phi); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        auto phi_arr = m_phi.array(mfi);
        auto ign_arr = ignition_mask->const_array(mfi);

        for (int j = box.smallEnd(1); j <= box.bigEnd(1); j++) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); i++) {
                if (ign_arr(i, j, 0) == 1) {
                    phi_arr(i, j, 0) = -1.0;
                    num_ignited++;
                }
            }
        }
    }

    ParallelDescriptor::ReduceIntSum(num_ignited);

    m_phi.FillBoundary(m_geom.periodicity());
    TagNarrowBand();

    {
        m_phi_window.Fill(m_phi);
        const LocalPhiWindow& nbr_phi_val = m_phi_window;

        const int dxd[4] = {1, -1, 0, 0};
        const int dyd[4] = {0, 0, 1, -1};

        for (int i = 0; i < m_narrow_band.GetNumNodes(); i++) {
            FireNodeData& nd = m_narrow_band[i];
            if (nd.burned && nd.time_of_ignition < 0.0) {
                bool is_perimeter = false;
                for (int d = 0; d < 4 && !is_perimeter; ++d)
                    if (nbr_phi_val(nd.ix + dxd[d], nd.iy + dyd[d]) >= 0.0)
                        is_perimeter = true;

                if (is_perimeter) {
                    nd.fuel_frac = 1.0;
                    nd.time_of_ignition = m_time;
                    if (m_fuel_db && m_fuel_db->HasFuelModel(nd.ifbfm)) {
                        FuelModelTable table = m_fuel_db->GetFuelModelTable(nd.ifbfm, nd.mlh);
                        nd.burnout_time_table = table.burnout_time;
                        nd.burnout_time       = table.burnout_time;
                        nd.flux_scaling       = 1.0;
                        if (m_use_source_fuel_scaling || m_use_grid_aware_burnout) {
                            const Real R_min = 1.0e-3;
                            Real tau_transit = m_cellsize / R_min;
                            Real tau_eff = table.burnout_time + tau_transit;
                            nd.burnout_time = tau_eff;
                            nd.flux_scaling = table.burnout_time / tau_eff;
                        }
                    }
                }
            }
        }
    }

    SaveNodeHistory();
    if (ParallelDescriptor::IOProcessor()) {
        Print() << "\n========================================\n";
        Print() << "AREA IGNITION FROM NETCDF MASK\n";
        Print() << "========================================\n";
        Print() << "Total ignited cells: " << num_ignited << "\n";
        Print() << "Area ignition applied successfully\n";
        Print() << "========================================\n\n";
    }
}


void ElmfireEngine::RestorePhiFromCheckpoint(const amrex::MultiFab& phi_restart)
{
    MultiFab::Copy(m_phi, phi_restart, 0, 0, 1, 0);
    m_phi.FillBoundary(m_geom.periodicity());

    SaveNodeHistory();
    TagNarrowBand();

    for (int i = 0; i < m_narrow_band.GetNumNodes(); i++) {
        FireNodeData& nd = m_narrow_band[i];
        if (nd.phi < 0.0 && nd.time_of_ignition < 0.0) {
            nd.burned = true;
            nd.fuel_frac = 0.0;
            nd.time_of_ignition = 0.0;
        }
    }
    SaveNodeHistory();

    if (ParallelDescriptor::IOProcessor()) {
        Print() << "Restored phi from checkpoint, rebuilt narrow band: "
                << m_narrow_band.GetNumNodes() << " active cells\n";
    }
}

// ============================================================================
// SyncPhiFromHierarchy: non-destructive multilevel phi sync (Phase 4)
// ============================================================================

void ElmfireEngine::SyncPhiFromHierarchy(const amrex::MultiFab& phi_src)
{
    int ngrow = std::min(m_phi.nGrow(), phi_src.nGrow());
    MultiFab::Copy(m_phi, phi_src, 0, 0, 1, ngrow);
    m_phi.FillBoundary(m_geom.periodicity());
    // No TagNarrowBand() call, no narrow-band node state touched — see
    // header comment / KS-HISTORY-001.
}

// ============================================================================
// ApplyValidPhiCorrection: monotone valid-phi update with node reconciliation
// ============================================================================

void ElmfireEngine::ApplyValidPhiCorrection(const amrex::MultiFab& candidate,
                                            amrex::Real sync_time,
                                            int edge_jlo, int edge_jhi)
{
    const Real tolerance = Real(1.0e-10) * std::max(Real(1.0), std::abs(sync_time));
    if (std::abs(m_time - sync_time) > tolerance) {
        Abort("ApplyValidPhiCorrection requires the engine at sync_time");
    }

    // Hierarchy MultiFabs may live on device memory. The persistent managed
    // scratch permits the existing host-side narrow-band bookkeeping on GPU
    // without an allocation on every coarse timestep.
    MultiFab::Copy(m_phi_correction, candidate, 0, 0, 1, 0);
    Gpu::streamSynchronize();

    m_last_phi_correction_stats = {};
    auto& stats = m_last_phi_correction_stats;
    const int nx = m_geom.Domain().length(0);
    std::unordered_set<int> existing_keys;
    existing_keys.reserve(m_narrow_band.GetNumNodes() * 2);
    for (int n = 0; n < m_narrow_band.GetNumNodes(); ++n) {
        const FireNodeData& nd = m_narrow_band[n];
        existing_keys.insert(nd.iy * nx + nd.ix);
    }
    auto tagged_keys = existing_keys;
    std::vector<std::pair<int, int>> newly_burned;
    std::unordered_set<int> transition_keys;

    for (MFIter mfi(m_phi); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        auto phi = m_phi.array(mfi);
        const auto correction = m_phi_correction.const_array(mfi);
        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                const Real old_phi = phi(i, j, 0);
                const Real new_phi = std::min(old_phi, correction(i, j, 0));
                if (new_phi < old_phi) {
                    const Real delta = old_phi - new_phi;
                    ++stats.changed_cells;
                    stats.total_phi_decrease += delta;
                    stats.max_phi_decrease = std::max(stats.max_phi_decrease, delta);
                }
                if (old_phi > 0.0 && new_phi <= 0.0) {
                    ++stats.newly_burned_cells;
                    if (j == edge_jlo || j == edge_jhi) {
                        ++stats.edge_newly_burned_cells;
                    }
                    newly_burned.emplace_back(i, j);
                    transition_keys.insert(j * nx + i);
                }
                phi(i, j, 0) = new_phi;
            }
        }
    }
    m_phi.FillBoundary(m_geom.periodicity());

    // TagBand checks valid-box ownership and retains all existing nodes.
    for (const auto& cell : newly_burned) {
        TagBand(cell.first, cell.second, tagged_keys);
    }

    stats.newly_tagged_nodes = m_narrow_band.GetNumNodes() -
                               static_cast<int>(existing_keys.size());
    for (int n = 0; n < m_narrow_band.GetNumNodes(); ++n) {
        FireNodeData& nd = m_narrow_band[n];
        const Real new_phi = GetPhiAt(nd.ix, nd.iy);
        nd.phi = new_phi;
        nd.phip_old = new_phi;
        nd.phip_rk1 = new_phi;

        const int key = nd.iy * nx + nd.ix;
        // Initialize history only for a sign transition caused by this
        // correction. Other negative nodes may have crossed zero during the
        // preceding coarse Advance and must retain that advance's history.
        if (transition_keys.count(key) &&
            (!existing_keys.count(key) || !nd.burned)) {
            ++stats.newly_burned_nodes;
            if (existing_keys.count(key)) ++stats.existing_nodes_activated;
            nd.burned = true;
            nd.fuel_frac = 1.0;
            nd.time_of_ignition = sync_time;
            nd.mlh = m_mlh;
            if (m_fuel_db && m_fuel_db->HasFuelModel(nd.ifbfm)) {
                const FuelModelTable table =
                    m_fuel_db->GetFuelModelTable(nd.ifbfm, nd.mlh);
                nd.burnout_time_table = table.burnout_time;
                nd.burnout_time = table.burnout_time;
                nd.flux_scaling = 1.0;
                if (m_use_grid_aware_burnout || m_use_source_fuel_scaling) {
                    const Real speed = std::sqrt(nd.ux * nd.ux + nd.uy * nd.uy);
                    const Real transit = m_cellsize / std::max(speed, Real(1.0e-3));
                    nd.burnout_time += transit;
                    nd.flux_scaling = table.burnout_time / nd.burnout_time;
                }
            }
            stats.new_node_fuel_frac_sum += nd.fuel_frac;
        }
    }
    SaveNodeHistory();
}

// ============================================================================
// FillGhostPhiFromHierarchy: ghost-only multilevel phi fill (Phase 4,
// narrowed scope, 2026-09-24)
// ============================================================================

void ElmfireEngine::FillGhostPhiFromHierarchy(const amrex::MultiFab& phi_src)
{
    // Both MultiFabs use this level's BoxArray and DistributionMapping.
    // Write only ghost cells, avoiding a per-substep valid-region snapshot
    // and three full MultiFab copies on the GPU.
    AMREX_ALWAYS_ASSERT(m_phi.boxArray() == phi_src.boxArray());
    AMREX_ALWAYS_ASSERT(m_phi.DistributionMap() == phi_src.DistributionMap());
    const int ngrow = std::min(m_phi.nGrow(), phi_src.nGrow());
    for (MFIter mfi(m_phi); mfi.isValid(); ++mfi) {
        const Box& valid = mfi.validbox();
        const Box copy_box = amrex::grow(valid, ngrow);
        const int ilo = valid.smallEnd(0), ihi = valid.bigEnd(0);
        const int jlo = valid.smallEnd(1), jhi = valid.bigEnd(1);
        const int klo = valid.smallEnd(2), khi = valid.bigEnd(2);
        const auto dst = m_phi.array(mfi);
        const auto src = phi_src.const_array(mfi);
        amrex::ParallelFor(copy_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (i < ilo || i > ihi || j < jlo || j > jhi || k < klo || k > khi) {
                dst(i,j,k) = src(i,j,k);
            }
        });
    }

    m_phi.FillBoundary(m_geom.periodicity());
    // No TagNarrowBand() call, no narrow-band node state touched — see
    // header comment / KS-HISTORY-001.
}

// ============================================================================
// REGRID: Redistribute MultiFabs and migrate narrow-band nodes
// ============================================================================

static constexpr int PACKED_NODE_SIZE = 57;

static void PackFireNode(const FireNodeData& node, double* buf)
{
    int k = 0;
    buf[k++] = static_cast<double>(node.ix);
    buf[k++] = static_cast<double>(node.iy);
    buf[k++] = static_cast<double>(node.ifbfm);
    buf[k++] = static_cast<double>(node.crown_fire);
    buf[k++] = static_cast<double>(node.burned ? 1 : 0);
    buf[k++] = static_cast<double>(node.front ? 1 : 0);
    buf[k++] = node.phi;
    buf[k++] = node.phip_old;
    buf[k++] = node.phip_rk1;
    buf[k++] = node.ux;
    buf[k++] = node.uy;
    buf[k++] = node.dphidx;
    buf[k++] = node.dphidy;
    buf[k++] = node.dphidx_limited;
    buf[k++] = node.dphidy_limited;
    buf[k++] = node.gradmag;
    buf[k++] = node.normx;
    buf[k++] = node.normy;
    buf[k++] = node.normx_dms;
    buf[k++] = node.normy_dms;
    buf[k++] = node.ws20;
    buf[k++] = node.wd20;
    buf[k++] = node.u_wind;
    buf[k++] = node.v_wind;
    buf[k++] = node.wsmf;
    buf[k++] = node.m1;
    buf[k++] = node.m10;
    buf[k++] = node.m100;
    buf[k++] = node.mlh;
    buf[k++] = node.mlw;
    buf[k++] = node.slope_x;
    buf[k++] = node.slope_y;
    buf[k++] = node.tanslp2;
    buf[k++] = node.vs0;
    buf[k++] = node.phiw_surface;
    buf[k++] = node.phis_surface;
    buf[k++] = node.velocity_dms_surface;
    buf[k++] = node.vback;
    buf[k++] = node.ir;
    buf[k++] = node.hpua_surface;
    buf[k++] = node.flin_dms_surface;
    buf[k++] = node.flame_length;
    buf[k++] = node.phisx;
    buf[k++] = node.phisy;
    buf[k++] = node.low;
    buf[k++] = node.adj;
    buf[k++] = node.b_coeff;
    buf[k++] = node.phiwterm;
    buf[k++] = node.phisterm;
    buf[k++] = node.uxousx;
    buf[k++] = node.uyousy;
    buf[k++] = node.time_added;
    buf[k++] = node.fuel_frac;
    buf[k++] = node.time_of_ignition;
    buf[k++] = node.burnout_time;
    buf[k++] = node.flin_at_ignition;
    buf[k++] = static_cast<double>(node.spot_emitter);
    AMREX_ASSERT(k == PACKED_NODE_SIZE);
}

static void UnpackFireNode(const double* buf, FireNodeData& node)
{
    int k = 0;
    node.ix         = static_cast<int>(buf[k++]);
    node.iy         = static_cast<int>(buf[k++]);
    node.ifbfm      = static_cast<int>(buf[k++]);
    node.crown_fire = static_cast<int>(buf[k++]);
    node.burned     = (static_cast<int>(buf[k++]) != 0);
    node.front      = (static_cast<int>(buf[k++]) != 0);
    node.phi            = buf[k++];
    node.phip_old       = buf[k++];
    node.phip_rk1       = buf[k++];
    node.ux             = buf[k++];
    node.uy             = buf[k++];
    node.dphidx         = buf[k++];
    node.dphidy         = buf[k++];
    node.dphidx_limited = buf[k++];
    node.dphidy_limited = buf[k++];
    node.gradmag        = buf[k++];
    node.normx          = buf[k++];
    node.normy          = buf[k++];
    node.normx_dms      = buf[k++];
    node.normy_dms      = buf[k++];
    node.ws20       = buf[k++];
    node.wd20       = buf[k++];
    node.u_wind     = buf[k++];
    node.v_wind     = buf[k++];
    node.wsmf       = buf[k++];
    node.m1         = buf[k++];
    node.m10        = buf[k++];
    node.m100       = buf[k++];
    node.mlh        = buf[k++];
    node.mlw        = buf[k++];
    node.slope_x    = buf[k++];
    node.slope_y    = buf[k++];
    node.tanslp2    = buf[k++];
    node.vs0                  = buf[k++];
    node.phiw_surface         = buf[k++];
    node.phis_surface         = buf[k++];
    node.velocity_dms_surface = buf[k++];
    node.vback                = buf[k++];
    node.ir                   = buf[k++];
    node.hpua_surface         = buf[k++];
    node.flin_dms_surface     = buf[k++];
    node.flame_length         = buf[k++];
    node.phisx      = buf[k++];
    node.phisy      = buf[k++];
    node.low        = buf[k++];
    node.adj        = buf[k++];
    node.b_coeff    = buf[k++];
    node.phiwterm   = buf[k++];
    node.phisterm   = buf[k++];
    node.uxousx     = buf[k++];
    node.uyousy     = buf[k++];
    node.time_added       = buf[k++];
    node.fuel_frac        = buf[k++];
    node.time_of_ignition = buf[k++];
    node.burnout_time     = buf[k++];
    node.flin_at_ignition = buf[k++];
    node.spot_emitter     = static_cast<int>(buf[k++]);
    AMREX_ASSERT(k == PACKED_NODE_SIZE);
}

static int FindOwnerRank(const amrex::BoxArray& ba,
                         const amrex::DistributionMapping& dm,
                         int ix, int iy)
{
    IntVect iv(AMREX_D_DECL(ix, iy, 0));
    for (int i = 0; i < static_cast<int>(ba.size()); ++i) {
        if (ba[i].contains(iv)) {
            return dm[i];
        }
    }
    return -1;
}

void ElmfireEngine::Regrid(const amrex::BoxArray& new_ba,
                            const amrex::DistributionMapping& new_dm)
{
    if (m_spot.enable) {
        Abort("Spotting (stage S1) supports a fixed fire level 0 only: regridding the fire engine is not supported");
    }
    if (m_extinction) {
        Abort("fire.extinction supports a fixed fire level 0 only: regridding the fire engine is not supported");
    }
    int myrank = ParallelDescriptor::MyProc();
    int nprocs = ParallelDescriptor::NProcs();

    int old_nodes = m_narrow_band.GetNumNodes();

    // Redistribute MultiFabs
    {
        int ngrow = m_phi.nGrow();
        MFInfo field_info;
        field_info.SetArena(The_Managed_Arena());
        MultiFab new_phi(new_ba, new_dm, 1, ngrow, field_info);
        new_phi.setVal(1.0);
        new_phi.ParallelCopy(m_phi, 0, 0, 1);
        std::swap(m_phi, new_phi);

        MultiFab new_correction(new_ba, new_dm, 1, 0, field_info);
        std::swap(m_phi_correction, new_correction);

        MultiFab new_fuel(new_ba, new_dm, 1, ngrow, field_info);
        new_fuel.setVal(static_cast<Real>(m_uniform_fuel_id));
        new_fuel.ParallelCopy(m_fuel_model, 0, 0, 1);
        std::swap(m_fuel_model, new_fuel);

        MultiFab new_history(new_ba, new_dm, NumFireHistoryComp, 0, field_info);
        new_history.setVal(0.0);
        new_history.setVal(1.0, FuelFrac, 1);
        new_history.setVal(-1.0, IgnitionTime, 1);
        new_history.setVal(600.0, BurnoutTime, 2);
        new_history.setVal(1.0, FluxScaling, 1);
        new_history.ParallelCopy(m_fire_history, 0, 0, NumFireHistoryComp);
        std::swap(m_fire_history, new_history);
    }

    m_phi.FillBoundary(m_geom.periodicity());
    m_fuel_model.FillBoundary(m_geom.periodicity());

    // Build per-destination send buffers
    std::vector<std::vector<double>> send_bufs(nprocs);

    for (int n = 0; n < m_narrow_band.GetNumNodes(); ++n) {
        const FireNodeData& nd = m_narrow_band[n];
        int dest = FindOwnerRank(new_ba, new_dm, nd.ix, nd.iy);
        if (dest >= 0) {
            send_bufs[dest].resize(send_bufs[dest].size() + PACKED_NODE_SIZE);
            PackFireNode(nd, &send_bufs[dest][send_bufs[dest].size() - PACKED_NODE_SIZE]);
        }
    }

    m_narrow_band.Clear();

    if (nprocs == 1) {
        int num_recv = static_cast<int>(send_bufs[0].size()) / PACKED_NODE_SIZE;
        for (int n = 0; n < num_recv; ++n) {
            FireNodeData& new_nd = m_narrow_band.AddNode(0, 0, 0.0);
            UnpackFireNode(&send_bufs[0][n * PACKED_NODE_SIZE], new_nd);
        }
    } else {
        std::vector<int> send_counts(nprocs);
        std::vector<int> send_displs(nprocs, 0);
        int total_send = 0;
        for (int p = 0; p < nprocs; ++p) {
            send_counts[p] = static_cast<int>(send_bufs[p].size());
            total_send += send_counts[p];
        }
        for (int p = 1; p < nprocs; ++p) {
            send_displs[p] = send_displs[p-1] + send_counts[p-1];
        }

        std::vector<double> send_data(total_send);
        for (int p = 0; p < nprocs; ++p) {
            if (!send_bufs[p].empty()) {
                std::copy(send_bufs[p].begin(), send_bufs[p].end(),
                          send_data.begin() + send_displs[p]);
            }
        }
        send_bufs.clear();

        std::vector<int> recv_counts(nprocs);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                     recv_counts.data(), 1, MPI_INT,
                     ParallelDescriptor::Communicator());

        std::vector<int> recv_displs(nprocs, 0);
        int total_recv = recv_counts[0];
        for (int p = 1; p < nprocs; ++p) {
            recv_displs[p] = recv_displs[p-1] + recv_counts[p-1];
            total_recv += recv_counts[p];
        }

        std::vector<double> recv_data(total_recv);
        MPI_Alltoallv(send_data.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE,
                      recv_data.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                      ParallelDescriptor::Communicator());

        int num_recv_nodes = total_recv / PACKED_NODE_SIZE;
        for (int n = 0; n < num_recv_nodes; ++n) {
            FireNodeData& new_nd = m_narrow_band.AddNode(0, 0, 0.0);
            UnpackFireNode(&recv_data[n * PACKED_NODE_SIZE], new_nd);
        }
    }

    if (ParallelDescriptor::IOProcessor()) {
        Print() << "ElmfireEngine::Regrid: " << old_nodes << " nodes before, "
                << m_narrow_band.GetNumNodes() << " nodes after migration\n";
    }
}


void ElmfireEngine::SetFuelModelSource(const std::string& source)
{
    m_fuel_model_source = source;
    Print() << "Set fuel model source: " << m_fuel_model_source << "\n";
}


static int IGBPToFuelModel(int igbp_index)
{
    switch (igbp_index) {
        case 1:  return 165;
        case 2:  return 145;
        case 3:  return 161;
        case 4:  return 161;
        case 5:  return 162;
        case 6:  return 124;
        case 7:  return 122;
        case 8:  return 123;
        case 9:  return 123;
        case 10: return 102;
        case 11: return 0;
        case 12: return 101;
        case 13: return 162;
        case 14: return 121;
        case 15: return 0;
        case 16: return 0;
        case 17: return 0;
        case 18: return 102;
        case 19: return 101;
        case 20: return 0;
        case 21: return 0;
        default: return 102;
    }
}


void ElmfireEngine::SetUniformFuel(int fuel_model_id)
{
    m_uniform_fuel_id = fuel_model_id;
    m_fuel_model.setVal(static_cast<Real>(fuel_model_id));
    Print() << "Set uniform fuel model: " << fuel_model_id << "\n";
}

void ElmfireEngine::SetUniformWeather(Real wind_speed,
                                      Real wind_direction,
                                      Real m1, Real m10, Real m100,
                                      Real mlh, Real mlw)
{
    constexpr Real WAF = 0.362;

    m_wind_speed = wind_speed * WAF;
    m_ws20_uniform = wind_speed;
    m_wind_direction = wind_direction;
    m_m1 = m1;
    m_m10 = m10;
    m_m100 = m100;
    m_mlh = mlh;
    m_mlw = mlw;

    Print() << "Weather set: Wind (20-ft)=" << wind_speed << " mph @ " << wind_direction << " deg\n";
    Print() << "  Wind Adjustment Factor (WAF) = " << WAF << "\n";
    Print() << "  Wind (midflame) = " << m_wind_speed << " mph\n";
    Print() << "  M1=" << m1 << ", M10=" << m10 << ", M100=" << m100 << "\n";
    Print() << "  MLH=" << mlh << ", MLW=" << mlw << "\n";
}

void ElmfireEngine::SetUniformSlope(Real slope_x, Real slope_y)
{
    m_slope_x = slope_x;
    m_slope_y = slope_y;
    m_tanslp2 = slope_x * slope_x + slope_y * slope_y;

    for (int i = 0; i < m_narrow_band.GetNumNodes(); i++) {
        FireNodeData& nd = m_narrow_band[i];
        nd.slope_x = m_slope_x;
        nd.slope_y = m_slope_y;
        nd.tanslp2 = m_tanslp2;
    }

    Real slope_magnitude = std::sqrt(m_tanslp2);
    Real slope_degrees = std::atan(slope_magnitude) * 180.0 / M_PI;

    Print() << "Slope set: slope_x=" << slope_x << ", slope_y=" << slope_y << "\n";
    Print() << "  Slope magnitude (rise/run): " << slope_magnitude << "\n";
    Print() << "  Slope angle: " << slope_degrees << " degrees\n";
    Print() << "  tanslp2 (for Rothermel): " << m_tanslp2 << "\n";
}

// ============================================================================
// MAIN ADVANCE
// ============================================================================

Real ElmfireEngine::Advance(Real dt)
{
    Real dt_cfl = ComputeTimestep();
    dt = std::min(dt, dt_cfl);

    if (ParallelDescriptor::IOProcessor()) {
        Print() << "\n";
        Print() << "Advancing from t=" << m_time << " with dt=" << dt << " s\n";
    }

    if (m_extinction) ExtinctionUpdate();

    if (m_spot.enable) SpotApplyArrivals(dt);

    UpdatePhysics();

    if (m_terrain_scale_factor != 1.0) {
        for (int i = 0; i < m_narrow_band.GetNumNodes(); i++) {
            m_narrow_band[i].phis_surface *= m_terrain_scale_factor;
        }
    }

    IntegrateRK2(dt);

    SyncNodesToMultiFab();
    SyncMultiFabToNodes();

    AddNewlyBurnedCells();

    if (m_spot.enable) SpotLaunch(m_time, dt);

    m_time += dt;
    ++m_step;

    UpdateFuelFraction();
    SaveNodeHistory();
    DetagBurnedOutCells();

    if (m_spot.enable) {
        long c[3] = {m_spot_launched, m_spot_landed, m_spot_ignitions};
        ParallelDescriptor::ReduceLongSum(c, 3);
        if (ParallelDescriptor::IOProcessor()) {
            Print() << "  Spotting (cumulative): embers launched " << c[0] << ", landed " << c[1]
                    << ", spot ignitions " << c[2] << "\n";
        }
    }

    int local_nodes = m_narrow_band.GetNumNodes();
    int global_nodes = local_nodes;
    ParallelDescriptor::ReduceIntSum(global_nodes);
    if (ParallelDescriptor::IOProcessor()) {
        Print() << "  Advanced to t=" << m_time << " s, narrow band: "
                << global_nodes << " nodes\n";
    }

    return dt;
}

Real ElmfireEngine::AdvanceInterval(Real fire_dt)
{
    Real t_elapsed = 0.0;
    int n_sub = 0;
    while (t_elapsed < fire_dt - 1.0e-8) {
        Real dt_used = Advance(fire_dt - t_elapsed);
        t_elapsed += dt_used;
        ++n_sub;
    }
    if (n_sub > 1 && ParallelDescriptor::IOProcessor()) {
        Print() << "  AdvanceInterval: fire_dt=" << fire_dt
                << " s completed in " << n_sub << " sub-steps\n";
    }
    return t_elapsed;
}

// ============================================================================
// Atmospheric Coupling Version
// ============================================================================

void ElmfireEngine::UpdateAtmosphericData(const amrex::MultiFab* wind_u,
                                         const amrex::MultiFab* wind_v,
                                         const amrex::MultiFab* terrain,
                                         const amrex::iMultiFab* fuel_map,
                                         const amrex::MultiFab* wind_height)
{
    ExtractAtmosphericWind(wind_u, wind_v, terrain, fuel_map, wind_height);
}

Real ElmfireEngine::Advance(Real dt,
                            const amrex::MultiFab* wind_u,
                            const amrex::MultiFab* wind_v,
                            const amrex::MultiFab* terrain,
                            const amrex::iMultiFab* fuel_map)
{
    ExtractAtmosphericWind(wind_u, wind_v, terrain, fuel_map);
    return Advance(dt);
}

// ============================================================================

void ElmfireEngine::ExtractAtmosphericWind(const amrex::MultiFab* wind_u,
                                           const amrex::MultiFab* wind_v,
                                           const amrex::MultiFab* terrain,
                                           const amrex::iMultiFab* fuel_map,
                                           const amrex::MultiFab* wind_height)
{
    using namespace amrex;

    // A per-cell wind height is read with the wind's MFIter, so it must share the wind layout
    if (wind_height != nullptr && wind_u != nullptr) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            wind_height->boxArray() == wind_u->boxArray() &&
            wind_height->DistributionMap() == wind_u->DistributionMap() &&
            wind_height->nGrowVect().allGE(wind_u->nGrowVect()),
            "ExtractAtmosphericWind: wind_height must match wind_u's BoxArray/DistributionMapping and ghost cells");
    }

    if (terrain != nullptr) m_has_terrain_slope = true;

    static bool fuel_multifab_initialized = false;
    if (!fuel_multifab_initialized) {
        if (ParallelDescriptor::IOProcessor()) {
            Print() << "  [ELMFIRE DEBUG] ExtractAtmosphericWind called. fuel_model_source='" << m_fuel_model_source
                    << "', fuel_map=" << (fuel_map ? "valid" : "nullptr") << "\n";
        }

        if (m_fuel_model_source == "landfire" && fuel_map != nullptr) {
            if (ParallelDescriptor::IOProcessor()) {
                Print() << "  [ELMFIRE] Using LANDFIRE fuel data (FBFM40)...\n";
            }

            int water_count = 0, urban_count = 0, grass_count = 0, forest_count = 0;
            for (MFIter mfi_dst(m_fuel_model); mfi_dst.isValid(); ++mfi_dst) {
                auto fuel_arr = m_fuel_model.array(mfi_dst);
                const Box& dst_box = mfi_dst.validbox();

                for (MFIter mfi_src(*fuel_map); mfi_src.isValid(); ++mfi_src) {
                    const Box overlap = dst_box & mfi_src.fabbox();
                    if (!overlap.ok()) continue;

                    auto landfire_arr = fuel_map->const_array(mfi_src);
                    for (int j = overlap.smallEnd(1); j <= overlap.bigEnd(1); ++j) {
                        for (int i = overlap.smallEnd(0); i <= overlap.bigEnd(0); ++i) {
                            int fuel_id = static_cast<int>(landfire_arr(i, j, 0));
                            fuel_arr(i, j, 0) = static_cast<Real>(fuel_id);

                            if (fuel_id >= 92 && fuel_id <= 99) water_count++;
                            else if (fuel_id == 91) urban_count++;
                            else if (fuel_id >= 101 && fuel_id <= 104) grass_count++;
                            else if (fuel_id >= 140) forest_count++;
                        }
                    }
                }
            }

            ParallelDescriptor::ReduceIntSum(water_count);
            ParallelDescriptor::ReduceIntSum(urban_count);
            ParallelDescriptor::ReduceIntSum(grass_count);
            ParallelDescriptor::ReduceIntSum(forest_count);

            if (ParallelDescriptor::IOProcessor()) {
                Print() << "  [ELMFIRE] LANDFIRE fuel data loaded: non-burnable(90-99)=" << water_count
                        << ", urban(91)=" << urban_count << ", grass(101-104)=" << grass_count
                        << ", forest(140+)=" << forest_count << "\n";
            }
        } else if (m_fuel_model_source == "metgrid_igbp" && fuel_map != nullptr) {
            if (ParallelDescriptor::IOProcessor()) {
                Print() << "  [ELMFIRE] Populating fuel MultiFab from IGBP land use data...\n";
            }

            int water_count = 0, urban_count = 0, grass_count = 0, forest_count = 0;
            for (MFIter mfi_dst(m_fuel_model); mfi_dst.isValid(); ++mfi_dst) {
                auto fuel_arr = m_fuel_model.array(mfi_dst);
                const Box& dst_box = mfi_dst.validbox();

                for (MFIter mfi_src(*fuel_map); mfi_src.isValid(); ++mfi_src) {
                    const Box overlap = dst_box & mfi_src.fabbox();
                    if (!overlap.ok()) continue;

                    auto lu_arr = fuel_map->const_array(mfi_src);
                    for (int j = overlap.smallEnd(1); j <= overlap.bigEnd(1); ++j) {
                        for (int i = overlap.smallEnd(0); i <= overlap.bigEnd(0); ++i) {
                            int igbp_index = static_cast<int>(lu_arr(i, j, 0));
                            int fuel_id = IGBPToFuelModel(igbp_index);
                            fuel_arr(i, j, 0) = static_cast<Real>(fuel_id);

                            if (fuel_id == 0) water_count++;
                            else if (fuel_id >= 101 && fuel_id <= 104) grass_count++;
                            else if (fuel_id >= 140) forest_count++;
                        }
                    }
                }
            }

            ParallelDescriptor::ReduceIntSum(water_count);
            ParallelDescriptor::ReduceIntSum(grass_count);
            ParallelDescriptor::ReduceIntSum(forest_count);
            ParallelDescriptor::ReduceIntSum(urban_count);
            if (ParallelDescriptor::IOProcessor()) {
                Print() << "  [ELMFIRE] Fuel MultiFab populated: water/urban(0)=" << water_count
                        << ", grass(101-104)=" << grass_count << ", forest(140+)=" << forest_count << "\n";
            }
        } else if (m_fuel_model_source == "uniform") {
            if (ParallelDescriptor::IOProcessor()) {
                Print() << "  [ELMFIRE] Populating fuel MultiFab with uniform fuel_id=" << m_uniform_fuel_id << "\n";
            }
            m_fuel_model.setVal(static_cast<Real>(m_uniform_fuel_id));
        } else {
            if (ParallelDescriptor::IOProcessor()) {
                Print() << "  [ELMFIRE WARNING] fuel_model_source not recognized or fuel_map=nullptr, MultiFab not populated!\n";
            }
        }

        fuel_multifab_initialized = true;
    }

    int num_nodes = m_narrow_band.GetNumNodes();

    int nodes_with_wind = 0;
    int nodes_without_wind = 0;

    if (num_nodes > 0) {

    for (int n = 0; n < num_nodes; n++) {
        FireNodeData& nd = m_narrow_band[n];

        int ix = nd.ix;
        int iy = nd.iy;

        bool wind_found = false;
        if (wind_u != nullptr && wind_v != nullptr) {
            for (MFIter mfi(*wind_u); mfi.isValid(); ++mfi) {
                const Box& box = mfi.fabbox();
                if (box.contains(IntVect(amrex::Dim3{ix, iy, 0}))) {
                    auto u_arr = wind_u->const_array(mfi);
                    auto v_arr = wind_v->const_array(mfi);

                    nd.u_wind = u_arr(ix, iy, 0);
                    nd.v_wind = v_arr(ix, iy, 0);
                    wind_found = true;

                    Real wspd_at_z = std::sqrt(nd.u_wind * nd.u_wind + nd.v_wind * nd.v_wind);
                    Real wdir = std::atan2(-nd.u_wind, -nd.v_wind) * 180.0 / M_PI;
                    if (wdir < 0.0) wdir += 360.0;

                    // Height above ground of the supplied wind (log-law conversion to 20 ft below)
                    Real z_extract = (wind_height != nullptr) ? wind_height->const_array(mfi)(ix, iy, 0)
                                                              : m_atmos_wind_height;

                    // Fuel model extraction
                    if ((m_fuel_model_source == "landfire" || m_fuel_model_source == "metgrid_igbp") && fuel_map != nullptr) {
                        for (MFIter mfi_fuel(*fuel_map); mfi_fuel.isValid(); ++mfi_fuel) {
                            const Box& fbox = mfi_fuel.fabbox();
                            if (fbox.contains(IntVect(amrex::Dim3{AMREX_D_DECL(ix, iy, 0})))) {
                                auto fuel_arr = fuel_map->const_array(mfi_fuel);

                                if (m_fuel_model_source == "landfire") {
                                    nd.ifbfm = static_cast<int>(fuel_arr(ix, iy, 0));
                                } else {
                                    int igbp_index = static_cast<int>(fuel_arr(ix, iy, 0));
                                    nd.ifbfm = IGBPToFuelModel(igbp_index);
                                }
                                break;
                            }
                        }
                    } else if (m_fuel_model_source == "uniform") {
                        nd.ifbfm = m_uniform_fuel_id;
                    }

                    // Slope from terrain
                    if (terrain != nullptr) {
                        for (MFIter mfi_t(*terrain); mfi_t.isValid(); ++mfi_t) {
                            const Box& tbox = mfi_t.fabbox();

                            if (tbox.contains(IntVect(amrex::Dim3{AMREX_D_DECL(ix-1, iy, 0}))) &&
                                tbox.contains(IntVect(amrex::Dim3{AMREX_D_DECL(ix+1, iy, 0}))) &&
                                tbox.contains(IntVect(amrex::Dim3{AMREX_D_DECL(ix, iy-1, 0}))) &&
                                tbox.contains(IntVect(amrex::Dim3{AMREX_D_DECL(ix, iy+1, 0})))) {

                                auto z_terrain = terrain->const_array(mfi_t);

                                Real z_east  = z_terrain(ix+1, iy, 0);
                                Real z_west  = z_terrain(ix-1, iy, 0);
                                Real z_north = z_terrain(ix, iy+1, 0);
                                Real z_south = z_terrain(ix, iy-1, 0);

                                auto dx_array = m_geom.CellSizeArray();
                                Real dx = dx_array[0];
                                Real dy = dx_array[1];

                                nd.slope_x = (z_east - z_west) / (2.0 * dx);
                                nd.slope_y = (z_north - z_south) / (2.0 * dy);
                                nd.tanslp2 = nd.slope_x * nd.slope_x + nd.slope_y * nd.slope_y;
                                break;
                            }
                        }
                    }

                    // Wind adjustment (log-law + WAF)
                    constexpr Real z_20ft = 6.1;
                    Real z0 = 0.03;
                    Real wind_adjustment = 0.3;

                    if (nd.ifbfm >= 101 && nd.ifbfm <= 109) {
                        z0 = 0.04; wind_adjustment = 0.42;
                    } else if (nd.ifbfm >= 121 && nd.ifbfm <= 124) {
                        z0 = 0.1; wind_adjustment = 0.38;
                    } else if (nd.ifbfm >= 141 && nd.ifbfm <= 149) {
                        z0 = 0.5; wind_adjustment = 0.27;
                    } else if (nd.ifbfm >= 161 && nd.ifbfm <= 165) {
                        z0 = 1.0; wind_adjustment = 0.32;
                    } else if (nd.ifbfm >= 181 && nd.ifbfm <= 189) {
                        z0 = 1.5; wind_adjustment = 0.11;
                    } else if (nd.ifbfm == 91) {
                        z0 = 1.0; wind_adjustment = 0.30;
                    }

                    if (z_extract <= z0) {
                        Abort("ExtractAtmosphericWind: wind height " + std::to_string(z_extract) +
                              " m is not above the roughness length " + std::to_string(z0) + " m");
                    }
                    Real log_ratio = std::log(z_20ft / z0) / std::log(z_extract / z0);
                    Real wspd_20ft = wspd_at_z * log_ratio;

                    Real effective_adjustment = wind_adjustment;
                    constexpr Real HIGH_WIND_THRESHOLD = 10.0;
                    if (wspd_20ft > HIGH_WIND_THRESHOLD) {
                        Real high_wind_multiplier = 1.0 + 0.4 * (wspd_20ft - HIGH_WIND_THRESHOLD) / 10.0;
                        high_wind_multiplier = std::min(high_wind_multiplier, 1.5);
                        effective_adjustment = std::min(wind_adjustment * high_wind_multiplier, 0.85);
                    }

                    Real wspd_midflame = wspd_20ft * effective_adjustment * m_wind_data_scale_factor;
                    constexpr Real MAX_MIDFLAME_WIND = 15.0;
                    wspd_midflame = std::min(wspd_midflame, MAX_MIDFLAME_WIND);

                    constexpr Real MS_TO_MPH = 2.23694;
                    nd.ws20 = wspd_20ft * MS_TO_MPH;
                    nd.wsmf = wspd_midflame * MS_TO_MPH;
                    nd.wd20 = wdir;

                    nodes_with_wind++;
                    wind_found = true;
                    break;
                }
            }
        }

        if (!wind_found) {
            nodes_without_wind++;
        }
    }
    } // end if (num_nodes > 0)

    int total_nodes_with_wind = nodes_with_wind;
    int total_nodes_without_wind = nodes_without_wind;
    ParallelDescriptor::ReduceIntSum(total_nodes_with_wind);
    ParallelDescriptor::ReduceIntSum(total_nodes_without_wind);

    if (m_time < 3.0 && ParallelDescriptor::IOProcessor()) {
        Print() << "Wind extraction summary at t=" << m_time << ":\n";
        Print() << "  Total nodes with wind: " << total_nodes_with_wind << "\n";
        Print() << "  Total nodes without wind: " << total_nodes_without_wind << "\n";
    }
}

// ============================================================================
// PRIVATE METHODS
// ============================================================================

void ElmfireEngine::SaveNodeHistory()
{
    for (MFIter mfi(m_fire_history); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        auto history = m_fire_history.array(mfi);
        for (int n = 0; n < m_narrow_band.GetNumNodes(); ++n) {
            const FireNodeData& nd = m_narrow_band[n];
            if (!box.contains(IntVect(amrex::Dim3{nd.ix, nd.iy, 0}))) continue;
            history(nd.ix, nd.iy, 0, FuelFrac) = nd.fuel_frac;
            history(nd.ix, nd.iy, 0, IgnitionTime) = nd.time_of_ignition;
            history(nd.ix, nd.iy, 0, BurnoutTime) = nd.burnout_time;
            history(nd.ix, nd.iy, 0, BurnoutTimeTable) = nd.burnout_time_table;
            history(nd.ix, nd.iy, 0, FluxScaling) = nd.flux_scaling;
            history(nd.ix, nd.iy, 0, Burned) = nd.burned ? 1.0 : 0.0;
            history(nd.ix, nd.iy, 0, CrownFire) = nd.crown_fire ? 1.0 : 0.0;
            history(nd.ix, nd.iy, 0, FlinAtIgnition) = nd.flin_at_ignition;
            history(nd.ix, nd.iy, 0, SpotEmitter) = nd.spot_emitter;
        }
    }
}

void ElmfireEngine::RestoreNodeHistory(FireNodeData& nd) const
{
    if (nd.phi >= 0.0) return;
    for (MFIter mfi(m_fire_history); mfi.isValid(); ++mfi) {
        if (!mfi.validbox().contains(IntVect(amrex::Dim3{nd.ix, nd.iy, 0}))) continue;
        auto history = m_fire_history.const_array(mfi);
        if (history(nd.ix, nd.iy, 0, Burned) > 0.5) {
            nd.fuel_frac = history(nd.ix, nd.iy, 0, FuelFrac);
            nd.time_of_ignition = history(nd.ix, nd.iy, 0, IgnitionTime);
            nd.burnout_time = history(nd.ix, nd.iy, 0, BurnoutTime);
            nd.burnout_time_table = history(nd.ix, nd.iy, 0, BurnoutTimeTable);
            nd.flux_scaling = history(nd.ix, nd.iy, 0, FluxScaling);
            nd.burned = true;
            nd.crown_fire = history(nd.ix, nd.iy, 0, CrownFire) > 0.5;
            nd.flin_at_ignition = history(nd.ix, nd.iy, 0, FlinAtIgnition);
            nd.spot_emitter = static_cast<int>(std::lround(history(nd.ix, nd.iy, 0, SpotEmitter)));
        }
        break;
    }
}

void ElmfireEngine::TagNarrowBand()
{
    SaveNodeHistory();
    m_narrow_band.Clear();

    const Box& domain = m_geom.Domain();
    int nx = domain.length(0);
    int ny = domain.length(1);

    std::unordered_set<int> tagged_set;

    // The narrow band can be wider than m_phi's solver stencil ghosts.
    // Rebuild from a temporary halo of the full physical band width so a
    // burned cell near an MPI rank boundary tags cells owned by both ranks.
    MFInfo band_info;
    band_info.SetArena(The_Managed_Arena());
    MultiFab band_phi(m_phi.boxArray(), m_phi.DistributionMap(), 1,
                      m_band_thickness, band_info);
    band_phi.setVal(1.0); // Nonperiodic physical/outside-level ghosts are unburned.
    MultiFab::Copy(band_phi, m_phi, 0, 0, 1, 0);
    band_phi.FillBoundary(m_geom.periodicity());

    for (MFIter mfi(band_phi); mfi.isValid(); ++mfi) {
        const Box scanbox = amrex::grow(mfi.validbox(), m_band_thickness) & domain;
        const auto& phi_arr = band_phi.const_array(mfi);

        for (int j = scanbox.smallEnd(1); j <= scanbox.bigEnd(1); ++j) {
            for (int i = scanbox.smallEnd(0); i <= scanbox.bigEnd(0); ++i) {
                Real phi_val = phi_arr(i, j, 0);

                if (phi_val <= 0.0) {
                    int istart = std::max(0, i - m_band_thickness);
                    int istop = std::min(nx - 1, i + m_band_thickness);
                    int jstart = std::max(0, j - m_band_thickness);
                    int jstop = std::min(ny - 1, j + m_band_thickness);

                    istart = std::max(istart, scanbox.smallEnd(0));
                    istop = std::min(istop, scanbox.bigEnd(0));
                    jstart = std::max(jstart, scanbox.smallEnd(1));
                    jstop = std::min(jstop, scanbox.bigEnd(1));

                    bool has_unburned_neighbor = false;
                    for (int jj = jstart; jj <= jstop; ++jj) {
                        for (int ii = istart; ii <= istop; ++ii) {
                            if (phi_arr(ii, jj, 0) > 0.0) {
                                has_unburned_neighbor = true;
                                break;
                            }
                        }
                        if (has_unburned_neighbor) break;
                    }

                    if (has_unburned_neighbor) {
                        TagBand(i, j, tagged_set);
                    }
                }
            }
        }
    }
}

void ElmfireEngine::TagBand(int ix_loc, int iy_loc, std::unordered_set<int>& tagged_set)
{
    const Box& domain = m_geom.Domain();
    int nx = domain.length(0);
    int ny = domain.length(1);

    int ixstart = std::max(0, ix_loc - m_band_thickness);
    int ixstop = std::min(nx - 1, ix_loc + m_band_thickness);
    int iystart = std::max(0, iy_loc - m_band_thickness);
    int iystop = std::min(ny - 1, iy_loc + m_band_thickness);

    for (int j = iystart; j <= iystop; ++j) {
        for (int i = ixstart; i <= ixstop; ++i) {
            int key = j * nx + i;
            if (tagged_set.count(key)) continue;

            bool is_owned_by_this_rank = false;
            Real phi_val = 1.0;

            for (MFIter mfi(m_phi); mfi.isValid(); ++mfi) {
                const Box& validbox = mfi.validbox();
                if (validbox.contains(IntVect(amrex::Dim3{i, j, 0}))) {
                    is_owned_by_this_rank = true;
                    const auto& phi_arr = m_phi.const_array(mfi);
                    phi_val = phi_arr(i, j, 0);
                    break;
                }
            }

            if (is_owned_by_this_rank) {
                FireNodeData& new_nd = m_narrow_band.AddNode(i, j, m_time);
                new_nd.phi = phi_val;
                new_nd.phip_old = phi_val;

                if (phi_val < 0.0) {
                    new_nd.burned = true;
                    new_nd.fuel_frac = 0.0;
                    new_nd.time_of_ignition = -1.0;
                }

                for (MFIter mfi_fuel(m_fuel_model); mfi_fuel.isValid(); ++mfi_fuel) {
                    const Box& fuel_box = mfi_fuel.validbox();
                    if (fuel_box.contains(IntVect(amrex::Dim3{i, j, 0}))) {
                        const auto& fuel_arr = m_fuel_model.const_array(mfi_fuel);
                        new_nd.ifbfm = static_cast<int>(fuel_arr(i, j, 0));
                        break;
                    }
                }

                RestoreNodeHistory(new_nd);
                tagged_set.insert(key);
            }
        }
    }
}

void ElmfireEngine::UpdatePhysics()
{
    int N = m_narrow_band.GetNumNodes();
    if (N == 0) return;

    // CPU: assign per-node scalar fields (fuel id, moisture, uniform wind, slope)
    for (int i = 0; i < N; i++) {
        FireNodeData& nd = m_narrow_band[i];

        if (m_fuel_model_source == "uniform" || m_fuel_model_source.empty())
            nd.ifbfm = m_uniform_fuel_id;

        nd.m1   = m_m1;
        nd.m10  = m_m10;
        nd.m100 = m_m100;
        nd.mlh  = m_mlh;
        nd.mlw  = m_mlw;

        if (m_wind_speed > 0.0 && nd.wsmf == 0.0) {
            nd.wsmf = m_wind_speed;
            nd.ws20 = m_ws20_uniform;   // only used by spotting
            nd.wd20 = m_wind_direction;
        }

        if (!m_has_terrain_slope) {
            nd.slope_x = m_slope_x;
            nd.slope_y = m_slope_y;
            nd.tanslp2 = m_tanslp2;
        }
    }

    // CPU: pre-compute FuelModelTable per node (database lookup cannot run on GPU)
    std::vector<FuelModelTable> tables_h(N);
    for (int i = 0; i < N; i++) {
        const FireNodeData& nd = m_narrow_band[i];
        if (nd.ifbfm != 0)
            tables_h[i] = m_fuel_db->GetFuelModelTable(nd.ifbfm, nd.mlh);
    }

    m_narrow_band.CopyToDevice();

#ifdef AMREX_USE_GPU
    amrex::Gpu::DeviceVector<FuelModelTable> tables_d;
    tables_d.resize(N);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          tables_h.begin(), tables_h.end(), tables_d.begin());
    amrex::Gpu::streamSynchronize();
    FuelModelTable* tables_dp = tables_d.data();
#else
    FuelModelTable* tables_dp = tables_h.data();
#endif

    FireNodeData* nodes_d = m_narrow_band.GetDevicePtr();

    LaunchSurfaceSpreadKernel(nodes_d, tables_dp, N);

    m_narrow_band.CopyFromDevice();
}

void ElmfireEngine::IntegrateRK2(Real& dt)
{
    m_phi.FillBoundary(m_geom.periodicity());

    constexpr Real FUEL_THRESHOLD = 0.01;
    const Box& domain = m_geom.Domain();
    const int nx     = domain.length(0);
    const int ilo    = domain.smallEnd(0);
    const int jlo    = domain.smallEnd(1);

    std::unordered_map<int, Real> burned_fuel_map;

    if (m_use_source_fuel_scaling) {
        burned_fuel_map.reserve(m_narrow_band.GetNumNodes());
        for (int i = 0; i < m_narrow_band.GetNumNodes(); i++) {
            const FireNodeData& nd = m_narrow_band[i];
            if (nd.burned)
                burned_fuel_map[(nd.iy - jlo) * nx + (nd.ix - ilo)] = nd.fuel_frac;
        }
        m_phi_window.Fill(m_phi);
    }

    const LocalPhiWindow& nbr_phi = m_phi_window;

    for (int istep = 1; istep <= 2; istep++) {
        const MultiFab* dead = m_dead_any ? &m_dead : nullptr;
        CalcNormalVectors(m_narrow_band, m_phi, 0.5 * m_rcellsize, istep, dead);

        int num_nodes = m_narrow_band.GetNumNodes();

        // CPU: pre-compute EllipticalFire tables (database lookup can't run on GPU)
        std::vector<FuelModelTable> ev_tables_h(num_nodes);
        for (int i = 0; i < num_nodes; i++) {
            const FireNodeData& nd = m_narrow_band[i];
            if (nd.ifbfm != 0)
                ev_tables_h[i] = m_fuel_db->GetFuelModelTable(nd.ifbfm, nd.mlh);
        }

        m_narrow_band.CopyToDevice();

#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceVector<FuelModelTable> ev_tables_d;
        ev_tables_d.resize(num_nodes);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                              ev_tables_h.begin(), ev_tables_h.end(), ev_tables_d.begin());
        amrex::Gpu::streamSynchronize();
        FuelModelTable* ev_tables_dp = ev_tables_d.data();
#else
        FuelModelTable* ev_tables_dp = ev_tables_h.data();
#endif

        FireNodeData* ev_nodes_d = m_narrow_band.GetDevicePtr();

        LaunchEllipticalFireKernel(ev_nodes_d, ev_tables_dp, num_nodes);

        m_narrow_band.CopyFromDevice();

        // CPU: source fuel scaling (uses phi_flat + burned_fuel_map, host-only)
        if (m_use_source_fuel_scaling) {
            for (int i = 0; i < num_nodes; i++) {
                FireNodeData& nd = m_narrow_band[i];
                if (nd.phi <= 0.0) continue;
                const int dxd[4] = {1, -1, 0, 0};
                const int dyd[4] = {0, 0, 1, -1};
                bool found_burned_nbr = false;
                Real sf = 0.0;
                for (int d = 0; d < 4 && sf < 1.0; ++d) {
                    int nbx = nd.ix + dxd[d];
                    int nby = nd.iy + dyd[d];
                    if (nbr_phi(nbx, nby) < 0.0) {
                        found_burned_nbr = true;
                        int key = (nby - jlo) * nx + (nbx - ilo);
                        auto it = burned_fuel_map.find(key);
                        sf = (it == burned_fuel_map.end())
                             ? Real(1.0)
                             : std::max(sf, Real(std::pow(it->second, m_source_fuel_exponent)));
                    }
                }
                if (!found_burned_nbr) sf = 1.0;
                nd.ux *= sf;
                nd.uy *= sf;
            }
        }

        CflAndFluxLimiter(m_narrow_band, m_phi, m_rcellsize, dt, istep, m_target_cfl, dead);
        Rk2Integrate(m_narrow_band, m_phi, dt, istep);
    }

    SyncNodesToMultiFab();
}

void ElmfireEngine::UpdateNarrowBand()
{
    TagNarrowBand();
}

void ElmfireEngine::AddNewlyBurnedCells()
{
    // IntegrateRK2 has just changed valid phi; refresh cross-rank ghosts
    // before hole filling and activation read neighboring cells.
    m_phi.FillBoundary(m_geom.periodicity());
    const Box& domain = m_geom.Domain();
    int nx = domain.length(0);

    // Hole-filling: forcibly ignite stranded unburned cells
    if (m_narrow_band.GetNumNodes() > 0) {
        m_phi_window.Fill(m_phi);
        const LocalPhiWindow& nbr_phi_hf = m_phi_window;

        const int dxd_hf[4] = {1,-1,0,0};
        const int dyd_hf[4] = {0,0,1,-1};
        bool any_filled = false;
        for (int i = 0; i < m_narrow_band.GetNumNodes(); i++) {
            FireNodeData& nd = m_narrow_band[i];
            if (!nd.burned && nd.phi > 0.0) {
                bool all_nbrs_burned = true;
                for (int d = 0; d < 4 && all_nbrs_burned; ++d)
                    if (nbr_phi_hf(nd.ix+dxd_hf[d], nd.iy+dyd_hf[d]) >= 0.0 ||
                        IsDead(nd.ix+dxd_hf[d], nd.iy+dyd_hf[d]))       // dead neighbours do not fill holes
                        all_nbrs_burned = false;
                if (all_nbrs_burned) {
                    nd.phi = -1.0;
                    any_filled = true;
                }
            }
        }
        if (any_filled) SyncNodesToMultiFab();
    }

    // Build O(1) lookup set from current narrow band
    std::unordered_set<int> tagged_set;
    tagged_set.reserve(m_narrow_band.GetNumNodes() * 2);
    for (int i = 0; i < m_narrow_band.GetNumNodes(); i++) {
        const FireNodeData& nd = m_narrow_band[i];
        tagged_set.insert(nd.iy * nx + nd.ix);
    }

    std::vector<std::pair<int, int>> newly_burned;

    int num_nodes = m_narrow_band.GetNumNodes();
    for (int i = 0; i < num_nodes; i++) {
        FireNodeData& nd = m_narrow_band[i];

        if (nd.phi <= 0.0 && !nd.burned) {
            nd.burned = true;
            nd.time_of_ignition = m_time;
            nd.flin_at_ignition = nd.flin_dms_surface;
            if (m_spot.enable) {
                const long key = static_cast<long>(nd.iy - domain.smallEnd(1)) * nx + (nd.ix - domain.smallEnd(0));
                nd.spot_emitter = SpotDecideEmitter(key, nd.flin_at_ignition);
            }

            if (m_fuel_db && m_fuel_db->HasFuelModel(nd.ifbfm)) {
                FuelModelTable table = m_fuel_db->GetFuelModelTable(nd.ifbfm, nd.mlh);
                nd.burnout_time_table = table.burnout_time;
                nd.burnout_time       = table.burnout_time;
                nd.flux_scaling       = 1.0;

                if (m_use_grid_aware_burnout || m_use_source_fuel_scaling) {
                    Real R_spread = std::sqrt(nd.ux * nd.ux + nd.uy * nd.uy);
                    const Real R_min = 1.0e-3;
                    Real R_eff = std::max(R_spread, R_min);
                    Real tau_transit = m_cellsize / R_eff;
                    nd.burnout_time = table.burnout_time + tau_transit;
                    nd.flux_scaling = table.burnout_time / nd.burnout_time;
                }
            }

            newly_burned.push_back({nd.ix, nd.iy});
        }
    }

    m_phi.FillBoundary(m_geom.periodicity());

    for (const auto& cell : newly_burned) {
        TagBand(cell.first, cell.second, tagged_set);
    }

    TagBandAroundBurnedGhosts(tagged_set);
}

void ElmfireEngine::TagBandAroundBurnedGhosts(std::unordered_set<int>& tagged_set)
{
    // Burned cells owned by other ranks (ghost cells, filled by the caller) activate this rank's
    // part of their band, as the owner does for its own cells.
    for (MFIter mfi(m_phi); mfi.isValid(); ++mfi) {
        const Box& validbox = mfi.validbox();
        const Box& ghostbox = mfi.growntilebox();
        const auto& phi_arr = m_phi.const_array(mfi);

        for (int j = ghostbox.smallEnd(1); j <= ghostbox.bigEnd(1); ++j) {
            for (int i = ghostbox.smallEnd(0); i <= ghostbox.bigEnd(0); ++i) {
                if (!validbox.contains(IntVect(amrex::Dim3{i, j, 0}))) {
                    if (phi_arr(i, j, 0) < 0.0) {
                        TagBand(i, j, tagged_set);
                    }
                }
            }
        }
    }
}

// ============================================================================
// SPOTTING (stage S1; docs/spotting_S1_design.md)
// ============================================================================

bool ElmfireEngine::SpotIgniteCell(int ix, int iy, std::unordered_set<int>& tagged_set)
{
    const IntVect iv(AMREX_D_DECL(ix, iy, 0));
    for (MFIter mfi(m_phi); mfi.isValid(); ++mfi) {
        if (!mfi.validbox().contains(iv)) continue;
        auto phi = m_phi.array(mfi);
        const int fuel = static_cast<int>(m_fuel_model.const_array(mfi)(ix, iy, 0));
        if (phi(ix, iy, 0) <= 0.0 || !Spotting::IsSpotIgnitable(m_spot, fuel)) return false;

        // Band first (phi still > 0, so the new nodes are unburned), then phi = -1: the cell is then
        // a newly burned cell for AddNewlyBurnedCells (ignition time, burnout, band, spotting).
        TagBand(ix, iy, tagged_set);
        for (int n = 0; n < m_narrow_band.GetNumNodes(); ++n) {
            FireNodeData& nd = m_narrow_band[n];
            if (nd.ix == ix && nd.iy == iy) {
                nd.phi = -1.0;
                nd.phip_old = -1.0;
                break;
            }
        }
        phi(ix, iy, 0) = -1.0;
        m_spot_ignited.array(mfi)(ix, iy, 0) = 1.0;
        ++m_spot_ignitions;
        m_spot_ignited_now.push_back(ix);
        m_spot_ignited_now.push_back(iy);
        return true;
    }
    return false;
}

void ElmfireEngine::ExtinctionUpdate()
{
    // Testing hook: at the first step with m_time >= test_time, burned cells inside test_box die (permanently).
    // Extinction criteria (local-condition fields) will be applied here.
    if (m_ext_test_box.size() != 4 || m_ext_test_done || m_time < m_ext_test_time) return;
    m_ext_test_done = true;
    const Real dx = m_cellsize;
    const Real* plo = m_geom.ProbLo();
    long ndead = 0;
    for (MFIter mfi(m_dead); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto d = m_dead.array(mfi);
        const auto p = m_phi.const_array(mfi);
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                const Real x = plo[0] + (i + 0.5) * dx, y = plo[1] + (j + 0.5) * dx;
                if (p(i, j, 0) <= 0.0 && x >= m_ext_test_box[0] && x <= m_ext_test_box[2] &&
                    y >= m_ext_test_box[1] && y <= m_ext_test_box[3]) {
                    d(i, j, 0) = 1.0; ++ndead;
                }
            }
    }
    m_dead.FillBoundary(m_geom.periodicity());
    m_dead_window.Fill(m_dead);
    ParallelDescriptor::ReduceLongSum(ndead);
    m_dead_any = true;
    Print() << "Extinction test: " << ndead << " burned cells died at t = " << m_time << " s\n";
}

void ElmfireEngine::SpotApplyArrivals(Real dt)
{
    const Box& domain = m_geom.Domain();
    const int nx = domain.length(0);
    const int ilo = domain.smallEnd(0), jlo = domain.smallEnd(1);

    std::unordered_set<int> tagged_set;
    tagged_set.reserve(2 * m_narrow_band.GetNumNodes() + 64);
    for (int n = 0; n < m_narrow_band.GetNumNodes(); ++n) {
        const FireNodeData& nd = m_narrow_band[n];
        tagged_set.insert(nd.iy * nx + nd.ix);
    }

    // Embers that arrived by the start of this step
    std::vector<SpotEmber> keep;
    keep.reserve(m_spot_pending.size());
    for (const SpotEmber& e : m_spot_pending) {
        if (e.t_arrive > m_time) { keep.push_back(e); continue; }
        ++m_spot_landed;
        const IntVect iv(AMREX_D_DECL(e.ix, e.iy, 0));
        for (MFIter mfi(m_ember_count); mfi.isValid(); ++mfi) {
            if (mfi.validbox().contains(iv)) {
                m_ember_count.array(mfi)(e.ix, e.iy, 0) += m_spot.ember_sampling_factor;
                break;
            }
        }
        if (m_spot.ignition == Spotting::Ignition::Direct) {
            if (e.positive) SpotIgniteCell(e.ix, e.iy, tagged_set);
        } else {
            const long key = static_cast<long>(e.iy - jlo) * nx + (e.ix - ilo);
            m_spot_deposited.emplace(key, Real(-1.0));
        }
    }
    m_spot_pending.swap(keep);

    // SIMPLE: per-step local ignition of cells holding embers, full ignition after the delay
    // (ELMFIRE EMBER_IGNITION). Keyed map -> fixed processing order.
    if (m_spot.ignition == Spotting::Ignition::Simple) {
        const Real p_ign = 0.01 * m_spot.pign;
        const Real p_n = 1.0 - std::pow(1.0 - p_ign, dt / std::max(m_spot.local_ignition_time, Real(1.0e-6)));
        for (auto it = m_spot_deposited.begin(); it != m_spot_deposited.end(); ) {
            const int ix = static_cast<int>(it->first % nx) + ilo;
            const int iy = static_cast<int>(it->first / nx) + jlo;
            Real phi_here = -1.0;
            const IntVect iv(AMREX_D_DECL(ix, iy, 0));
            for (MFIter mfi(m_phi); mfi.isValid(); ++mfi) {
                if (mfi.validbox().contains(iv)) { phi_here = m_phi.const_array(mfi)(ix, iy, 0); break; }
            }
            if (phi_here <= 0.0) { it = m_spot_deposited.erase(it); continue; }   // burned meanwhile
            if (it->second < 0.0) {
                if (Spotting::Uniform(m_spot.seed, it->first, m_step, 0, Spotting::kSimpleIgnition) < p_n) {
                    it->second = m_time + dt;
                }
                ++it;
            } else if (m_time + dt - it->second >= m_spot.cell_ignition_delay) {
                SpotIgniteCell(ix, iy, tagged_set);
                it = m_spot_deposited.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Spot fires near a box edge: every rank adds its part of their band now, as the owner did for
    // its cells, so the result does not depend on the domain decomposition (collective).
    const int nprocs = ParallelDescriptor::NProcs();
    if (nprocs > 1) {
        int nloc = static_cast<int>(m_spot_ignited_now.size());
        std::vector<int> counts(nprocs), displs(nprocs, 0);
        MPI_Allgather(&nloc, 1, MPI_INT, counts.data(), 1, MPI_INT, ParallelDescriptor::Communicator());
        for (int r = 1; r < nprocs; ++r) displs[r] = displs[r - 1] + counts[r - 1];
        std::vector<int> all(displs[nprocs - 1] + counts[nprocs - 1]);
        MPI_Allgatherv(m_spot_ignited_now.data(), nloc, MPI_INT, all.data(), counts.data(), displs.data(),
                       MPI_INT, ParallelDescriptor::Communicator());
        for (std::size_t k = 0; k + 1 < all.size(); k += 2) {
            TagBand(all[k], all[k + 1], tagged_set);   // only this rank's cells; no-op where already tagged
        }
    }
    m_spot_ignited_now.clear();
}

void ElmfireEngine::SpotLaunch(Real t0, Real dt)
{
    const Box& domain = m_geom.Domain();
    const int nx = domain.length(0), ny = domain.length(1);
    const int ilo = domain.smallEnd(0), jlo = domain.smallEnd(1);
    const Real dx = m_cellsize;
    const Spotting::Params& p = m_spot;
    constexpr Real kPi = 3.14159265358979323846;

    std::vector<double> sendbuf;   // 4 doubles per ember: ix, iy, t_arrive, positive
    const int N = m_narrow_band.GetNumNodes();
    for (int n = 0; n < N; ++n) {
        FireNodeData& nd = m_narrow_band[n];
        if (!nd.burned || nd.time_of_ignition < 0.0) continue;
        if (IsDead(nd.ix, nd.iy)) continue;                 // extinguished cells do not emit
        const Real t_start = nd.time_of_ignition;
        const Real overlap = std::min(t0 + dt, t_start + p.tau_embergen) - std::max(t0, t_start);
        if (overlap <= 1.0e-5) continue;

        const long key = static_cast<long>(nd.iy - jlo) * nx + (nd.ix - ilo);
        const std::uint64_t step = static_cast<std::uint64_t>(m_step);
        if (nd.spot_emitter < 0) {
            // Burned by an area/point/line ignition, not by spread: its fire-line intensity is the one
            // computed this step (ELMFIRE: initially burned cells emit too).
            nd.flin_at_ignition = nd.flin_dms_surface;
            nd.spot_emitter = SpotDecideEmitter(key, nd.flin_at_ignition);
        }
        if (p.per_step_draws) {
            if (!(nd.flin_at_ignition > p.critical_flin)) continue;   // ELMFIRE: FLIN > critical
            if (100.0 * Spotting::Uniform(p.seed, key, step, 0, Spotting::kEmitter) >= p.surface_spotting_pct) continue;
        } else if (!nd.spot_emitter) {
            continue;
        }

        // Number of (sampled) embers this step
        Real n_real = 0.0;
        if (p.generation == Spotting::Generation::Random) {
            const bool first_step = (t0 <= t_start + 1.0e-9);
            if (p.per_step_draws || first_step) {
                const std::uint64_t s = p.per_step_draws ? step : 0;
                n_real = p.nembers_min + Spotting::Uniform(p.seed, key, s, 0, Spotting::kCount)
                                         * (p.nembers_max - p.nembers_min);
            }
        } else {
            n_real = Spotting::EmberRate(p, nd.flin_at_ignition, dx, overlap);
        }
        const Real n_s = n_real / p.ember_sampling_factor;
        int nemb = static_cast<int>(std::floor(n_s));
        if (Spotting::Uniform(p.seed, key, step, 0, Spotting::kCountFrac) <= n_s - nemb) ++nemb;
        if (nemb <= 0) continue;

        const Real ws = 0.447 * nd.ws20;             // 20-ft wind [m/s]
        if (ws < 0.01) continue;                     // no transport (ELMFIRE drops these)
        Real mu, sigma, mu_cw, sigma_cw;
        Spotting::DistanceParameters(p, nd.ws20, nd.flin_at_ignition, mu, sigma, mu_cw, sigma_cw);

        m_spot_launched += nemb;
        for (int e = 0; e < nemb; ++e) {
            const std::uint64_t ue = static_cast<std::uint64_t>(e);
            Real dist = Spotting::SampleDistance(p, mu, sigma, dx,
                                                 Spotting::Uniform(p.seed, key, step, ue, Spotting::kDistance));
            Real travel = dist / ws;
            if (travel > p.tstop) { travel = p.tstop; dist = ws * p.tstop; }   // lands where it is at TSTOP
            const Real eps  = 8.0 * (Spotting::Uniform(p.seed, key, step, ue, Spotting::kDirection) - 0.5);
            const Real wdto = (nd.wd20 + 180.0 + eps) * kPi / 180.0;         // downwind, deg from north
            const Real ux = std::sin(wdto), uy = std::cos(wdto);
            Real x = (nd.ix - ilo + 0.5) * dx + dist * ux;
            Real y = (nd.iy - jlo + 0.5) * dx + dist * uy;
            if (p.use_crosswind) {
                const Real dev = std::sqrt(2.0) * Spotting::ErfInv(
                    2.0 * Spotting::Uniform(p.seed, key, step, ue, Spotting::kCrosswind) - 1.0) * sigma_cw + mu_cw;
                x += -uy * dev;   // 90 deg to the left of the trajectory, as in ELMFIRE
                y +=  ux * dev;
            }
            const int ixr = static_cast<int>(std::floor(x / dx));
            const int iyr = static_cast<int>(std::floor(y / dx));
            if (ixr <= 0 || ixr >= nx - 1 || iyr <= 0 || iyr >= ny - 1) continue;   // left the domain
            int positive = 0;
            if (p.ignition == Spotting::Ignition::Direct) {
                const Real ignprob = 0.01 * p.pign * p.source_fuel_ign_mult;
                positive = (Spotting::Uniform(p.seed, key, step, ue, Spotting::kDirectIgnition) < ignprob &&
                            dist > 0.5 * dx) ? 1 : 0;
            }
            if (!p.dump_embers.empty()) {
                if (!m_spot_dump) {
                    char fn[64];
                    std::snprintf(fn, sizeof(fn), "_rank%04d.csv", ParallelDescriptor::MyProc());
                    m_spot_dump = std::make_unique<std::ofstream>(p.dump_embers + fn);
                    *m_spot_dump << "t_launch,ix_from,iy_from,ix_to,iy_to,x_to_m,y_to_m,distance_m,t_arrive,positive,flin_kwm,ws20_mph,wd20_deg\n";
                }
                *m_spot_dump << t0 << ',' << nd.ix << ',' << nd.iy << ',' << (ixr + ilo) << ',' << (iyr + jlo) << ','
                             << x << ',' << y << ',' << dist << ',' << (t0 + travel) << ',' << positive << ',' << nd.flin_at_ignition << ','
                             << nd.ws20 << ',' << nd.wd20 << '\n';
            }
            sendbuf.push_back(ixr + ilo);
            sendbuf.push_back(iyr + jlo);
            sendbuf.push_back(t0 + travel);
            sendbuf.push_back(positive);
        }
    }

    // Exchange: every rank receives all landings and keeps those in its own boxes
    std::vector<double> all;
    const int nprocs = ParallelDescriptor::NProcs();
    if (nprocs > 1) {
        int nloc = static_cast<int>(sendbuf.size());
        std::vector<int> counts(nprocs), displs(nprocs, 0);
        MPI_Allgather(&nloc, 1, MPI_INT, counts.data(), 1, MPI_INT, ParallelDescriptor::Communicator());
        for (int r = 1; r < nprocs; ++r) displs[r] = displs[r - 1] + counts[r - 1];
        all.resize(displs[nprocs - 1] + counts[nprocs - 1]);
        MPI_Allgatherv(sendbuf.data(), nloc, MPI_DOUBLE, all.data(), counts.data(), displs.data(),
                       MPI_DOUBLE, ParallelDescriptor::Communicator());
    } else {
        all.swap(sendbuf);
    }
    for (std::size_t k = 0; k + 3 < all.size(); k += 4) {
        const int ix = static_cast<int>(all[k]), iy = static_cast<int>(all[k + 1]);
        const IntVect iv(AMREX_D_DECL(ix, iy, 0));
        for (MFIter mfi(m_phi); mfi.isValid(); ++mfi) {
            if (mfi.validbox().contains(iv)) {
                m_spot_pending.push_back({ix, iy, static_cast<Real>(all[k + 2]), static_cast<int>(all[k + 3])});
                break;
            }
        }
    }
}

int ElmfireEngine::SpotDecideEmitter(long key, Real flin) const
{
    // Per-cell draw (step size independent); with per_step_draws the draw happens every step in SpotLaunch
    if (m_spot.per_step_draws) return 1;
    const double u = Spotting::Uniform(m_spot.seed, key, 0, 0, Spotting::kEmitter);
    return (flin > m_spot.critical_flin && 100.0 * u < m_spot.surface_spotting_pct) ? 1 : 0;
}

Real ElmfireEngine::MaxSpottingDistance() const
{
    Real dmax = 0.0;
    if (m_spot.enable) {
        for (int n = 0; n < m_narrow_band.GetNumNodes(); ++n) {
            const FireNodeData& nd = m_narrow_band[n];
            if (nd.burned || !(nd.flin_dms_surface > m_spot.critical_flin)) continue;
            dmax = std::max(dmax, Spotting::MaxDistance(m_spot, nd.ws20, nd.flin_dms_surface, m_cellsize));
        }
    }
    ParallelDescriptor::ReduceRealMax(dmax);
    return dmax;
}

void ElmfireEngine::UpdateFuelFraction()
{
    for (int i = 0; i < m_narrow_band.GetNumNodes(); i++) {
        FireNodeData& nd = m_narrow_band[i];
        if (nd.burned && nd.time_of_ignition >= 0.0) {
            Real elapsed = m_time - nd.time_of_ignition;
            if (nd.burnout_time > 0.0) {
                nd.fuel_frac = std::max(0.0, 1.0 - elapsed / nd.burnout_time);
            } else {
                nd.fuel_frac = 0.0;
            }
        }
    }
}

void ElmfireEngine::DetagBurnedOutCells()
{
    constexpr Real BURNOUT_THRESHOLD = 0.01;
    constexpr Real PHI_FAR_THRESHOLD = 50.0;

    m_phi.FillBoundary(m_geom.periodicity());

    const Box& domain = m_geom.Domain();
    int nx = domain.length(0);

    // Build set of cells to remove
    std::unordered_set<int> to_remove;
    int removed_burnedout = 0;
    int removed_farfront  = 0;

    for (int i = 0; i < m_narrow_band.GetNumNodes(); i++) {
        const FireNodeData& nd = m_narrow_band[i];
        int ix = nd.ix, iy = nd.iy;

        if (nd.burned && nd.fuel_frac < BURNOUT_THRESHOLD) {
            bool near_front = false;
            for (MFIter mfi(m_phi); mfi.isValid() && !near_front; ++mfi) {
                const Box& fbox = mfi.fabbox();
                // Only the owning valid box may decide detagging. A ghost
                // box can omit part of the band and changes with MPI layout.
                if (mfi.validbox().contains(IntVect(amrex::Dim3{ix, iy, 0}))) {
                    const auto& phi_arr = m_phi.const_array(mfi);
                    int r = m_band_thickness;
                    int istart = std::max(fbox.smallEnd(0), ix - r);
                    int istop  = std::min(fbox.bigEnd(0),   ix + r);
                    int jstart = std::max(fbox.smallEnd(1), iy - r);
                    int jstop  = std::min(fbox.bigEnd(1),   iy + r);
                    for (int jj = jstart; jj <= jstop && !near_front; ++jj)
                        for (int ii = istart; ii <= istop && !near_front; ++ii)
                            if (phi_arr(ii, jj, 0) >= 0.0) near_front = true;
                    break;
                }
            }
            if (!near_front) {
                to_remove.insert(iy * nx + ix);
                removed_burnedout++;
            }
        } else if (!nd.burned && nd.phi > PHI_FAR_THRESHOLD) {
            to_remove.insert(iy * nx + ix);
            removed_farfront++;
        }
    }

    if (!to_remove.empty()) {
        m_narrow_band.RemoveIf([&](const FireNodeData& nd) {
            return to_remove.count(nd.iy * nx + nd.ix) > 0;
        });
    }

    int removed = removed_burnedout + removed_farfront;
    if (removed > 0 && ParallelDescriptor::IOProcessor()) {
        static int detag_print_count = 0;
        if (detag_print_count < 20 || removed > 100) {
            Print() << "  Detagged " << removed_burnedout << " burned-out + "
                    << removed_farfront << " far-front cells, "
                    << m_narrow_band.GetNumNodes() << " nodes remaining\n";
            detag_print_count++;
        }
    }
}

void ElmfireEngine::SyncNodesToMultiFab()
{
    for (MFIter mfi(m_phi); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto& phi_arr = m_phi.array(mfi);

        for (int i = 0; i < m_narrow_band.GetNumNodes(); i++) {
            const FireNodeData& nd = m_narrow_band[i];
            if (box.contains(IntVect(amrex::Dim3{nd.ix, nd.iy, 0}))) {
                phi_arr(nd.ix, nd.iy, 0) = nd.phi;
            }
        }
    }
}

void ElmfireEngine::SyncMultiFabToNodes()
{
    for (MFIter mfi(m_phi); mfi.isValid(); ++mfi) {
        const auto& phi_arr = m_phi.const_array(mfi);
        const Box& box = mfi.validbox();

        for (int i = 0; i < m_narrow_band.GetNumNodes(); i++) {
            FireNodeData& nd = m_narrow_band[i];
            // A node belongs to exactly one valid box. Reading another box's
            // ghost can overwrite its new phi with a stale pre-step value;
            // the result depends on MPI box ownership and MFIter order.
            if (box.contains(IntVect(amrex::Dim3{nd.ix, nd.iy, 0}))) {
                nd.phi = phi_arr(nd.ix, nd.iy, 0);
            }
        }
    }
}

Real ElmfireEngine::GetPhiAt(int ix, int iy) const
{
    for (MFIter mfi(m_phi); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        if (box.contains(IntVect(amrex::Dim3{ix, iy, 0}))) {
            const auto& phi_arr = m_phi.const_array(mfi);
            return phi_arr(ix, iy, 0);
        }
    }
    return 1.0;
}

Real ElmfireEngine::ComputeTimestep() const
{
    Real u_max = 0.0;
    int num_nodes = m_narrow_band.GetNumNodes();

    for (int i = 0; i < num_nodes; i++) {
        const FireNodeData& nd = m_narrow_band[i];
        Real u_mag = std::max(std::abs(nd.ux), std::abs(nd.uy));
        if (u_mag > u_max) u_max = u_mag;
    }

    ParallelDescriptor::ReduceRealMax(u_max);

    if (u_max > 1.0e-10) {
        return m_target_cfl * m_cellsize / u_max;
    } else {
        return 100.0;
    }
}

void ElmfireEngine::GetVelocityField(MultiFab& velocity_u,
                                     MultiFab& velocity_v) const
{
    velocity_u.setVal(0.0);
    velocity_v.setVal(0.0);

    for (MFIter mfi(velocity_u); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto& u_arr = velocity_u.array(mfi);
        const auto& v_arr = velocity_v.array(mfi);

        for (int i = 0; i < m_narrow_band.GetNumNodes(); i++) {
            const FireNodeData& nd = m_narrow_band[i];
            if (box.contains(IntVect(amrex::Dim3{nd.ix, nd.iy, 0}))) {
                u_arr(nd.ix, nd.iy, 0) = nd.ux;
                v_arr(nd.ix, nd.iy, 0) = nd.uy;
            }
        }
    }
}

void ElmfireEngine::GetTaggedCells(MultiFab& tagged) const
{
    tagged.setVal(0.0);

    for (MFIter mfi(tagged); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        auto tagged_arr = tagged.array(mfi);

        for (int i = 0; i < m_narrow_band.GetNumNodes(); i++) {
            const FireNodeData& nd = m_narrow_band[i];
            if (box.contains(IntVect(amrex::Dim3{nd.ix, nd.iy, 0}))) {
                tagged_arr(nd.ix, nd.iy, 0) = 1.0;
            }
        }
    }
}
