#include "FireHierarchy.H"
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_TagBox.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_FabArrayUtility.H>
#include <AMReX_GpuContainers.H>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>

using namespace amrex;

namespace {
void RestrictMinFeedbackKernel(MultiFab& coarse_blocks,
                               const MultiFab& fine,
                               int min_burned, bool sign_only)
{
    for (MFIter mfi(coarse_blocks, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto dst = coarse_blocks.array(mfi);
        auto src = fine.const_array(mfi);
        const Box bx = mfi.tilebox();
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const int fi = 2 * i;
            const int fj = 2 * j;
            const int fk = 2 * k;
            Real value = src(fi,fj,fk);
            int burned = 0;
            for (int dz = 0; dz < 2; ++dz) {
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const Real fine_phi = src(fi+dx,fj+dy,fk+dz);
                        value = amrex::min(value, fine_phi);
                        if (dz == 0 && fine_phi <= 0.0) ++burned;
                    }
                }
            }
            dst(i,j,k) = burned >= min_burned
                ? (sign_only ? Real(0.0) : value) : Real(1.0e30);
        });
    }
}
}

namespace {
struct FirePhiBoundaryFill {
    AMREX_GPU_DEVICE
    void operator()(const IntVect&, Array4<Real> const&, int, int,
                    GeometryData const&, Real, const BCRec*, int, int) const noexcept {}
};

void RecordArrivalTimeKernel(MultiFab& arrival_mf, const MultiFab& old_mf,
                             const MultiFab& new_mf, Real time, Real dt)
{
    for (MFIter mfi(arrival_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto arrival = arrival_mf.array(mfi);
        auto old_phi = old_mf.const_array(mfi);
        auto new_phi = new_mf.const_array(mfi);
        const Box bx = mfi.tilebox();
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (arrival(i,j,k) < 0.0 && new_phi(i,j,k) <= 0.0) {
                const Real old_value = old_phi(i,j,k);
                const Real new_value = new_phi(i,j,k);
                const Real fraction = old_value > 0.0 && old_value != new_value
                    ? amrex::min(Real(1.0), amrex::max(Real(0.0),
                        old_value / (old_value - new_value)))
                    : Real(0.0);
                arrival(i,j,k) = time + dt * fraction;
            }
        });
    }
}

void SetArrivalCellKernel(MultiFab& arrival_mf, int ix, int iy, Real time)
{
    for (MFIter mfi(arrival_mf); mfi.isValid(); ++mfi) {
        if (mfi.validbox().contains(IntVect(amrex::Dim3{ix,iy,0}))) {
            auto arrival = arrival_mf.array(mfi);
            const IntVect cell(amrex::Dim3{ix,iy,0});
            ParallelFor(Box(cell, cell),
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    arrival(i,j,k) = time;
                });
        }
    }
}

void FillValidMaskKernel(MultiFab& out, const iMultiFab& valid, int mask_comp)
{
    for (MFIter mfi(out, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto dst = out.array(mfi);
        auto src = valid.const_array(mfi);
        const Box bx = mfi.tilebox();
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            dst(i,j,k,mask_comp) = static_cast<Real>(src(i,j,k));
        });
    }
}
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

FireHierarchy::FireHierarchy()
{
    ConstructCommon();
}

FireHierarchy::FireHierarchy(const Geometry& level0_geom, const AmrInfo& amr_info)
    : AmrCore(level0_geom, amr_info), m_coupled(true)
{
    ConstructCommon();
}

void FireHierarchy::ConstructCommon()
{
    ReadParameters();

    int nlevs_max = max_level + 1;
    m_phi_new.resize(nlevs_max);
    m_phi_old.resize(nlevs_max);
    m_arrival_time.resize(nlevs_max);
    m_phi_ghost_filled.resize(nlevs_max);
    m_phi_min_coarsened.resize(nlevs_max);
    m_phi_min_candidate.resize(nlevs_max);
    m_t_new.resize(nlevs_max, 0.0);
    m_t_old.resize(nlevs_max, -1.0e10);
    m_fire.resize(nlevs_max);
    m_istep.resize(nlevs_max, 0);

    // Subcycling: each level takes ref_ratio substeps per coarser step
    m_nsubsteps.resize(nlevs_max, 1);
    for (int lev = 1; lev < nlevs_max; ++lev) {
        m_nsubsteps[lev] = MaxRefRatio(lev - 1);
    }

    // Boundary conditions: extrapolation (first-order) for phi
    m_bcrec.resize(1);  // 1 component
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        m_bcrec[0].setLo(idim, BCType::foextrap);
        m_bcrec[0].setHi(idim, BCType::foextrap);
    }
}

FireHierarchy::~FireHierarchy() = default;

// ============================================================================
// ReadParameters
// ============================================================================

void FireHierarchy::ReadParameters()
{
    // AMR parameters (standard AmrCore reads amr.*; in coupled mode amr.* belongs to the host)
    if (!m_coupled) {
        ParmParse pp("amr");
        pp.query("regrid_int", m_regrid_int);
    }

    // Spotting (stage S1) runs on a single, fixed fire level
    {
        bool spotting = false;
        ParmParse("fire.spotting").query("enable", spotting);
        if (spotting && max_level > 0) {
            Abort("fire.spotting (stage S1) supports fire level 0 only: set the fire max_level to 0");
        }
        bool extinction = false;
        ParmParse("fire.extinction").query("enable", extinction);
        if (extinction && max_level > 0) {
            Abort("fire.extinction supports fire level 0 only: set the fire max_level to 0");
        }
    }

    // Fire-specific AMR parameters
    {
        ParmParse pp("fire_amr");
        pp.query("phi_buffer", m_phi_buffer);
        pp.query("fuel_frac_threshold", m_fuel_frac_threshold);

        // Dynamic AMR is unvalidated since Phase 4 and out of scope (Phase 14), so a
        // multilevel run must choose its mode explicitly instead of silently getting it.
        // Single-level runs (max_level=0) keep the previous internal default.
        std::string mode_str = "dynamic";
        if (!pp.query("refinement_mode", mode_str) && max_level > 0) {
            Abort("fire_amr.refinement_mode must be set explicitly when amr.max_level > 0 "
                  "('static' is the supported mode; 'dynamic' is unvalidated)");
        }
        if (mode_str == "static") {
            m_refinement_mode = RefinementMode::Static;
        } else if (mode_str == "dynamic") {
            m_refinement_mode = RefinementMode::Dynamic;
        } else {
            Abort("fire_amr.refinement_mode must be 'dynamic' or 'static' (combined mode is Phase 14, not yet implemented), got: " + mode_str);
        }

        pp.query("static_corridor_y0", m_static_corridor_y0);
        pp.query("static_corridor_width", m_static_corridor_width);
        pp.query("static_corridor_amplitude", m_static_corridor_amplitude);
        pp.query("static_corridor_wavelength", m_static_corridor_wavelength);
        pp.query("static_polyline_file", m_static_polyline_file);
        pp.query("static_mask_file", m_static_mask_file);
        if (!m_static_mask_file.empty()) {
            if (m_refinement_mode != RefinementMode::Static || max_level != 1) {
                Abort("fire_amr.static_mask_file requires static refinement with amr.max_level=1");
            }
            if (!m_static_polyline_file.empty()) {
                Abort("Specify either fire_amr.static_mask_file or static_polyline_file, not both");
            }
            std::ifstream mask(m_static_mask_file, std::ios::binary);
            if (!mask) Abort("Cannot open static refinement mask: " + m_static_mask_file);
            std::string magic, dims_line, grid_line;
            if (!std::getline(mask, magic) || magic != "LSFIRE_REFINEMENT_MASK_V1" ||
                !std::getline(mask, dims_line) || !std::getline(mask, grid_line)) {
                Abort("Invalid static refinement mask header: " + m_static_mask_file);
            }
            int nx = 0, ny = 0;
            Real x0 = 0.0, y0 = 0.0, dx = 0.0, dy = 0.0;
            std::string extra;
            std::istringstream dims(dims_line), grid(grid_line);
            if (!(dims >> nx >> ny) || (dims >> extra) ||
                !(grid >> x0 >> y0 >> dx >> dy) || (grid >> extra)) {
                Abort("Invalid static refinement mask dimensions or geometry");
            }
            const auto& geom = Geom(0);
            const auto& domain = geom.Domain();
            const auto problo = geom.ProbLoArray();
            const auto cellsize = geom.CellSizeArray();
            const auto close = [](Real a, Real b) {
                return std::isfinite(a) && std::isfinite(b) &&
                    std::abs(a-b) <= Real(1.e-10) * std::max(Real(1.0), std::abs(b));
            };
            if (nx != domain.length(0) || ny != domain.length(1) ||
                !close(x0, problo[0]) || !close(y0, problo[1]) ||
                !close(dx, cellsize[0]) || !close(dy, cellsize[1])) {
                Abort("Static refinement mask does not match L0 grid geometry");
            }
            m_static_mask_nx = nx;
            m_static_refine_mask.resize(static_cast<std::size_t>(nx) * ny);
            mask.read(reinterpret_cast<char*>(m_static_refine_mask.data()),
                      static_cast<std::streamsize>(m_static_refine_mask.size()));
            if (mask.gcount() != static_cast<std::streamsize>(m_static_refine_mask.size()) ||
                mask.peek() != std::char_traits<char>::eof()) {
                Abort("Static refinement mask payload size is incorrect");
            }
            if (std::any_of(m_static_refine_mask.begin(), m_static_refine_mask.end(),
                            [](unsigned char v) { return v > 1; }) ||
                std::none_of(m_static_refine_mask.begin(), m_static_refine_mask.end(),
                             [](unsigned char v) { return v == 1; })) {
                Abort("Static refinement mask must contain 0/1 bytes and at least one tag");
            }
        }
        if (!m_static_polyline_file.empty()) {
            if (m_refinement_mode != RefinementMode::Static) {
                Abort("fire_amr.static_polyline_file requires static refinement");
            }
            std::ifstream line_file(m_static_polyline_file);
            if (!line_file) Abort("Cannot open static polyline file: " + m_static_polyline_file);
            std::string line;
            Real old_x = 0.0, old_y = 0.0;
            bool have_old = false;
            while (std::getline(line_file, line)) {
                if (line.empty() || line[0] == '#' || line == "x_m,y_m") continue;
                std::replace(line.begin(), line.end(), ',', ' ');
                std::istringstream row(line);
                Real x, y;
                if (!(row >> x >> y) || !std::isfinite(x) || !std::isfinite(y)) {
                    Abort("Invalid x_m,y_m row in static polyline file: " + line);
                }
                if (have_old) {
                    if (x == old_x && y == old_y) Abort("Static polyline has a zero-length segment");
                    m_static_polyline_segments.insert(m_static_polyline_segments.end(),
                                                      {old_x, old_y, x, y});
                }
                old_x = x;
                old_y = y;
                have_old = true;
            }
            if (m_static_polyline_segments.empty()) {
                Abort("Static polyline file requires at least two points");
            }
        }
        if (m_static_polyline_file.empty() && m_static_corridor_amplitude != 0.0 &&
            m_static_corridor_wavelength <= 0.0) {
            Abort("fire_amr.static_corridor_wavelength must be positive when amplitude is nonzero");
        }
        pp.query("static_min_feedback", m_static_min_feedback);
        pp.query("static_preserve_band_width", m_static_preserve_band_width);
        std::string phi_interpolation = "conservative";
        pp.query("phi_interpolation", phi_interpolation);
        if (phi_interpolation == "bilinear") {
            m_phi_bilinear_interp = true;
        } else if (phi_interpolation != "conservative") {
            Abort("fire_amr.phi_interpolation must be conservative or bilinear");
        }
        if (m_phi_bilinear_interp && m_refinement_mode != RefinementMode::Static) {
            Abort("fire_amr.phi_interpolation=bilinear requires static refinement");
        }
        if (m_static_preserve_band_width && m_refinement_mode != RefinementMode::Static) {
            Abort("fire_amr.static_preserve_band_width requires static refinement");
        }
        pp.query("static_min_feedback_min_burned", m_static_min_feedback_min_burned);
        pp.query("static_min_feedback_sign_only", m_static_min_feedback_sign_only);
        pp.query("feedback_diagnostic_file", m_feedback_diagnostic_file);
        pp.query("feedback_diagnostic_interval", m_feedback_diagnostic_interval);
        if (m_feedback_diagnostic_interval < 1) {
            Abort("fire_amr.feedback_diagnostic_interval must be positive");
        }
        if (m_static_min_feedback_min_burned < 1 || m_static_min_feedback_min_burned > 4) {
            Abort("fire_amr.static_min_feedback_min_burned must be 1..4");
        }
        if (m_static_min_feedback && m_refinement_mode != RefinementMode::Static) {
            Abort("fire_amr.static_min_feedback requires static refinement");
        }
    }

    // Domain / time / output parameters: standalone driver only. A coupled host decides
    // when to stop (AdvanceTo), so ComputeDt must not clip to a standalone stop time.
    if (m_coupled) {
        m_stop_time = std::numeric_limits<Real>::max();
    }
    if (!m_coupled) {
        ParmParse pp("time");
        pp.query("t_final", m_stop_time);
        pp.query("output_interval", m_plot_interval);
        pp.query("max_step", m_max_step);
        pp.query("dt", m_dt_coarse);   // outer AMR step (0 = auto from CFL)
    }

    if (!m_coupled) {
        ParmParse pp("output");
        pp.query("output_dir", m_output_dir);
    }

    // Fire model parameters
    {
        ParmParse pp("fire");

        pp.query("cfl", m_fire_config.cfl);
        pp.query("band_thickness", m_fire_config.band_thickness);
        pp.query("fuel_model_source", m_fire_config.fuel_model_source);
        pp.query("uniform_fuel_id", m_fire_config.uniform_fuel_id);
        pp.query("fuel_database_path", m_fire_config.fuel_database_path);

        m_fire_config.use_constant_wind = true;
        pp.query("constant_wind_u", m_fire_config.wind_u);
        pp.query("constant_wind_v", m_fire_config.wind_v);
        pp.query("wind_height", m_fire_config.wind_height);

        pp.query("fuel_moisture_1hr", m_fire_config.m1);
        pp.query("fuel_moisture_10hr", m_fire_config.m10);
        pp.query("fuel_moisture_100hr", m_fire_config.m100);
        pp.query("fuel_moisture_live_herb", m_fire_config.mlh);
        pp.query("fuel_moisture_live_woody", m_fire_config.mlw);

        pp.query("uniform_slope_x", m_fire_config.slope_x);
        pp.query("uniform_slope_y", m_fire_config.slope_y);

        pp.query("wind_data_scale_factor", m_fire_config.wind_data_scale_factor);

        // Fire suppression options (default: off for backward compatibility)
        pp.query("use_grid_aware_burnout",  m_fire_config.use_grid_aware_burnout);
        pp.query("use_source_fuel_scaling", m_fire_config.use_source_fuel_scaling);
        pp.query("source_fuel_exponent",    m_fire_config.source_fuel_exponent);
        pp.query("terrain_scale_factor",    m_fire_config.terrain_scale_factor);

        pp.query("fire_dt", m_fire_dt);    // fire advance interval (0 = use outer step)

        pp.query("ignition_x", m_ignition_x);
        pp.query("ignition_y", m_ignition_y);
        pp.query("ignition_radius", m_ignition_radius);
        pp.query("ignition_time", m_ignition_time);
        pp.query("ignition_shape", m_ignition_shape);
        if (m_ignition_shape != "point" && m_ignition_shape != "line") {
            Abort("fire.ignition_shape must be point or line");
        }
        if (m_ignition_shape == "line") {
            pp.get("ignition_line_xlo", m_ignition_line_xlo);
            pp.get("ignition_line_xhi", m_ignition_line_xhi);
            pp.get("ignition_line_width", m_ignition_line_width);
            if (m_ignition_line_xhi <= m_ignition_line_xlo ||
                m_ignition_line_width <= 0.0 || m_ignition_time > 0.0) {
                Abort("line ignition requires xhi > xlo, positive width, and ignition_time=0");
            }
        }
    }

    m_cfl = m_fire_config.cfl;
}

// ============================================================================
// InitData
// ============================================================================

void FireHierarchy::ValidateRefinementSources() const
{
    if (m_refinement_mode == RefinementMode::Static && m_static_refine_mask.empty() &&
        m_static_corridor_width <= 0.0) {
        Abort("fire_amr.refinement_mode=static requires fire_amr.static_corridor_width > 0 "
              "or a refinement mask");
    }
}

void FireHierarchy::SetStaticRefineMask(const std::vector<unsigned char>& mask, int nx, int ny)
{
    if (m_refinement_mode != RefinementMode::Static || max_level != 1) {
        Abort("SetStaticRefineMask requires fire_amr.refinement_mode=static and max_level=1");
    }
    if (!m_static_mask_file.empty() || !m_static_polyline_file.empty()) {
        Abort("SetStaticRefineMask: fire_amr.static_mask_file/static_polyline_file already given");
    }
    const Box& domain = Geom(0).Domain();
    if (nx != domain.length(0) || ny != domain.length(1) ||
        mask.size() != static_cast<std::size_t>(nx) * ny) {
        Abort("SetStaticRefineMask: mask dimensions do not match the level-0 fire grid");
    }
    if (std::any_of(mask.begin(), mask.end(), [](unsigned char v) { return v > 1; }) ||
        std::none_of(mask.begin(), mask.end(), [](unsigned char v) { return v == 1; })) {
        Abort("SetStaticRefineMask: mask must contain 0/1 values and at least one 1");
    }
    m_static_refine_mask = mask;
    m_static_mask_nx = nx;
}

void FireHierarchy::InitData()
{
    ValidateRefinementSources();

    // AmrCore::InitFromScratch creates level 0, then calls regrid to
    // potentially add finer levels based on ErrorEst tagging.
    InitFromScratch(0.0);

    // Now run initial regrid to create fine level at ignition point
    // (ErrorEst will tag ignition location for refinement)
    if (max_level > 0) {
        regrid(0, 0.0);
    }

    FinishInitData();
}

void FireHierarchy::InitData(const BoxArray& ba0, const DistributionMapping& dm0)
{
    ValidateRefinementSources();
    const Real time = 0.0;

    // Level 0 on the prescribed grids (as AmrMesh::MakeNewGrids(time), minus MakeBaseGrids)
    finest_level = 0;
    MakeNewLevelFromScratch(0, time, ba0, dm0);
    SetBoxArray(0, ba0);
    SetDistributionMap(0, dm0);

    // Finer levels from ErrorEst tags, one level at a time (as AmrMesh::MakeNewGrids)
    if (max_level > 0) {
        Vector<BoxArray> new_grids(max_level+1);
        new_grids[0] = grids[0];
        while (finest_level < max_level) {
            int new_finest;
            MakeNewGrids(finest_level, time, new_finest, new_grids);
            if (new_finest <= finest_level) { break; }
            finest_level = new_finest;
            DistributionMapping dm = MakeDistributionMap(new_finest, new_grids[new_finest]);
            MakeNewLevelFromScratch(new_finest, time, new_grids[new_finest], dm);
            SetBoxArray(new_finest, new_grids[new_finest]);
            SetDistributionMap(new_finest, dm);
        }
        regrid(0, time);   // same follow-up as InitData()
    }

    FinishInitData();
}

void FireHierarchy::FinishInitData()
{
    // Create a fire engine on EVERY level (Phase 1: per-level FireEngine).
    // Each level runs the same FireCore physics independently on its own
    // grid; there is no coarse/fine coupling yet (introduced in later
    // multilevel phases — see docs/LSFIRE_static_multilevel_plan.md).
    for (int lev = 0; lev <= finest_level; ++lev) {
        if (!m_fire[lev]) {
            const BoxArray& ba = boxArray(lev);
            const DistributionMapping& dm = DistributionMap(lev);
            const Geometry& gm = Geom(lev);

            ELMFIRE::Config level_config = m_fire_config;
            if (m_static_preserve_band_width) {
                for (int l = 1; l <= lev; ++l) {
                    level_config.band_thickness *= MaxRefRatio(l - 1);
                }
            }
            m_fire[lev] = std::make_unique<ELMFIRE::FireEngine>();
            m_fire[lev]->Initialize(gm, level_config, &ba, &dm);
        }
    }

    // Set ignition on every level (same physical location/time everywhere)
    if (m_ignition_time <= 0.0) {
        for (int lev = 0; lev <= finest_level; ++lev) {
            IgniteLevel(lev, 0.0, true);
        }
        m_ignited = true;
    }

    // Create output directory (standalone driver; a coupled host writes its own output)
    if (!m_coupled && ParallelDescriptor::IOProcessor()) {
        std::string cmd = "mkdir -p " + m_output_dir;
        std::system(cmd.c_str());
    }

    // Initialize cumulative flux accumulators at L0 (fixed grid, doesn't change with regrid)
    m_cum_heat_flux.define(boxArray(0), DistributionMap(0), 1, 0);
    m_cum_moist_flux.define(boxArray(0), DistributionMap(0), 1, 0);
    m_cum_heat_flux.setVal(0.0);
    m_cum_moist_flux.setVal(0.0);

    // Write initial state
    if (!m_coupled) { WritePlotFile(0); }

    Print() << "\nElmfireAmrCore initialized:\n"
            << "  max_level = " << max_level << "\n"
            << "  finest_level = " << finest_level << "\n"
            << "  regrid_int = " << m_regrid_int << "\n"
            << "  phi_buffer = " << m_phi_buffer << " m\n";
    for (int lev = 0; lev <= finest_level; ++lev) {
        Print() << "  Level " << lev << ": "
                << boxArray(lev).size() << " boxes, "
                << "dx = " << Geom(lev).CellSize(0) << " m\n";
    }
    Print() << "\n";
}

// ============================================================================
// IgniteLevel: ignition source on one level (overridable, e.g. area ignition)
// ============================================================================

void FireHierarchy::SetIgnitionMask(int lev, const iMultiFab* mask)
{
    if (lev >= static_cast<int>(m_ignition_masks.size())) { m_ignition_masks.resize(lev + 1, nullptr); }
    m_ignition_masks[lev] = mask;
}

void FireHierarchy::IgniteLevel(int lev, amrex::Real time, bool initial)
{
    // Area ignition from a mask: every cell of the mask ignites at `time`
    if (lev < static_cast<int>(m_ignition_masks.size()) && m_ignition_masks[lev]) {
        m_fire[lev]->SetAreaIgnition(m_ignition_masks[lev]);
        m_fire[lev]->GetPhiField(m_phi_new[lev]);
        for (MFIter mfi(m_arrival_time[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto arrival = m_arrival_time[lev].array(mfi);
            auto phi = m_phi_new[lev].const_array(mfi);
            ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                if (arrival(i,j,k) < Real(0.0) && phi(i,j,k) <= Real(0.0)) { arrival(i,j,k) = time; }
            });
        }
        return;
    }

    if (initial) {
        if (m_ignition_shape == "line") {
            amrex::MFInfo mask_info;
            mask_info.SetArena(amrex::The_Managed_Arena());
            iMultiFab ignition_mask(boxArray(lev), DistributionMap(lev), 1, 0, mask_info);
            ignition_mask.setVal(0);
            const Geometry& gm = Geom(lev);
            const Real xlo = m_ignition_line_xlo;
            const Real xhi = m_ignition_line_xhi;
            const Real ylo = m_ignition_y - 0.5 * m_ignition_line_width;
            const Real yhi = m_ignition_y + 0.5 * m_ignition_line_width;
            const Real* problo = gm.ProbLo();
            const Real* dx = gm.CellSize();
            for (MFIter mfi(ignition_mask); mfi.isValid(); ++mfi) {
                auto mask = ignition_mask.array(mfi);
                const Box& box = mfi.validbox();
                for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                    const Real y = problo[1] + (j + 0.5) * dx[1];
                    if (y < ylo || y >= yhi) continue;
                    for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                        const Real x = problo[0] + (i + 0.5) * dx[0];
                        if (x >= xlo && x < xhi) mask(i, j, 0) = 1;
                    }
                }
            }
            m_fire[lev]->SetAreaIgnition(&ignition_mask);
        } else {
            m_fire[lev]->SetIgnition(m_ignition_x, m_ignition_y,
                                      m_ignition_radius, 0.0);
        }
        // Copy phi from each level's own fire engine to its AMR field
        m_fire[lev]->GetPhiField(m_phi_new[lev]);
        for (MFIter mfi(m_arrival_time[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto arrival = m_arrival_time[lev].array(mfi);
            auto phi = m_phi_new[lev].const_array(mfi);
            const Box bx = mfi.tilebox();
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                if (phi(i,j,k) <= 0.0) arrival(i,j,k) = Real(0.0);
            });
        }
    } else {
        m_fire[lev]->SetIgnition(m_ignition_x, m_ignition_y,
                                m_ignition_radius, time);
        // Point ignition is an event at `time`, not a crossing
        // inferred from the following numerical advance.
        const auto plo = Geom(lev).ProbLoArray();
        const auto dx = Geom(lev).CellSizeArray();
        const int ix = static_cast<int>((m_ignition_x - plo[0]) / dx[0]);
        const int iy = static_cast<int>((m_ignition_y - plo[1]) / dx[1]);
        SetArrivalCellKernel(m_arrival_time[lev], ix, iy, time);
    }
}

// ============================================================================
// Evolve (main time loop)
// ============================================================================

void FireHierarchy::Evolve()
{
    Print() << "========================================\n";
    Print() << "Starting AMR simulation\n";
    Print() << "========================================\n\n";

    Real cur_time = m_t_new[0];
    Real next_plot_time = cur_time + m_plot_interval;
    int coarse_step = 0;

    while (cur_time < m_stop_time && coarse_step < m_max_step) {
        // Compute coarse dt
        Real dt = ComputeDt(0);
        if (cur_time + dt > m_stop_time) {
            dt = m_stop_time - cur_time;
        }

        // Advance all levels (recursive subcycling)
        timeStep(0, cur_time, dt);

        cur_time += dt;
        coarse_step++;

        // Output
        if (cur_time >= next_plot_time || coarse_step == m_max_step) {
            int fire_lev = FireLevel();
            Real fire_area = GetCompositeBurnedArea();
            int nb_size = m_fire[fire_lev] ? m_fire[fire_lev]->GetNarrowBandSize() : 0;

            Print() << "Step " << coarse_step
                    << ": t = " << cur_time << " s"
                    << ", composite_fire_area = " << fire_area << " m^2"
                    << ", narrow_band = " << nb_size
                    << ", levels = " << finest_level + 1
                    << "\n";



            m_plot_count++;
            WritePlotFile(m_plot_count);
            next_plot_time += m_plot_interval;
        }
    }

    Print() << "\n========================================\n";
    Print() << "AMR simulation complete!\n";
    Print() << "  Steps: " << coarse_step << "\n";
    Print() << "  Final time: " << cur_time << " s\n";
    Print() << "  Plot files: " << m_plot_count << "\n";
    Print() << "========================================\n";
}

// ============================================================================
// AdvanceTo (coupled driver: advance all levels to a target time)
// ============================================================================

int FireHierarchy::AdvanceTo(amrex::Real t_end)
{
    Real cur_time = m_t_new[0];
    int nsteps = 0;
    // Relative tolerance so a host step is never followed by a round-off sliver step
    const Real eps = Real(1.e-12) * std::max(Real(1.0), std::abs(t_end));
    while (cur_time < t_end - eps) {
        Real dt = ComputeDt(0);
        if (cur_time + dt > t_end - eps) {
            dt = t_end - cur_time;
        }
        // Land a step on fire.ignition_time so a delayed ignition happens exactly then
        // (ignition is applied at the start of the first step with time >= ignition_time)
        if (!m_ignited && m_ignition_time > cur_time + eps && cur_time + dt > m_ignition_time + eps) {
            dt = m_ignition_time - cur_time;
        }
        timeStep(0, cur_time, dt);
        cur_time += dt;
        ++nsteps;
    }
    return nsteps;
}

// ============================================================================
// timeStep (recursive subcycling)
// ============================================================================

void FireHierarchy::timeStep(int lev, amrex::Real time, amrex::Real dt)
{
    if (m_refinement_mode == RefinementMode::Static) {
        const Real tolerance = 1.0e-10 * std::max(Real(1.0), std::abs(time));
        if (std::abs(m_t_new[lev] - time) > tolerance) {
            Abort("Static fire level is out of sync with its parent timestep");
        }
    }

    // Regrid at this level.
    // Static refinement mode: the corridor is fixed at init (InitData's
    // initial regrid()) and never re-tagged from the fire front, so periodic
    // regridding is skipped entirely here — see docs/LSFIRE_static_multilevel_plan.md
    // Phase 2 ("no runtime regridding"). fire_amr.regrid_int is ignored in
    // this mode.
    if (m_refinement_mode == RefinementMode::Dynamic &&
        m_regrid_int > 0 && lev < max_level) {
        if (m_istep[lev] > 0 && m_istep[lev] % m_regrid_int == 0) {
            regrid(lev, time);

            // regrid(lev, time) may create/remake any level in (lev, finest_level].
            // Existing engines persist via RemakeLevel()'s Regrid() call (Option 2).
            // Handle the edge case where MakeNewLevelFromCoarse created a level
            // with no prior fire engine (e.g., level was removed then re-added) —
            // check every level that could have been touched, not just one.
            for (int l = lev + 1; l <= finest_level; ++l) {
                if (!m_fire[l]) {
                    const BoxArray& ba = boxArray(l);
                    const DistributionMapping& dm = DistributionMap(l);
                    const Geometry& gm = Geom(l);

                    m_fire[l] = std::make_unique<ELMFIRE::FireEngine>();
                    m_fire[l]->Initialize(gm, m_fire_config, &ba, &dm);
                    m_fire[l]->RestorePhiFromCheckpoint(m_phi_new[l]);
                    if (m_ignited) {
                        m_fire[l]->SetTime(time);
                    }
                }
            }
        }
    }

    // Save old state
    MultiFab::Copy(m_phi_old[lev], m_phi_new[lev], 0, 0, 1, 0);
    m_t_old[lev] = m_t_new[lev];

    // Advance this level
    AdvanceLevel(lev, time, dt);
    m_t_new[lev] = time + dt;
    m_istep[lev]++;

    // Recurse to finer levels (subcycling)
    if (lev < finest_level) {
        const Real fine_dt = dt / m_nsubsteps[lev + 1];
        for (int i = 0; i < m_nsubsteps[lev + 1]; ++i) {
            const Real fine_time = time + i * fine_dt;
            timeStep(lev + 1, fine_time, fine_dt);
        }

        // Experimental minimum restriction after fine subcycling. The old
        // independent path remains the default for controlled A/B runs.
        if (m_static_min_feedback && m_fire[lev]) {
            if (lev != 0 || finest_level != 1) {
                Abort("Experimental static feedback currently supports L0/L1 only");
            }
            MinDownPhi(lev);
            const Box covered = amrex::coarsen(boxArray(lev+1).minimalBox(),
                                               refRatio(lev));
            m_fire[lev]->ApplyValidPhiCorrection(m_phi_min_candidate[lev],
                                                 m_t_new[lev],
                                                 covered.smallEnd(1),
                                                 covered.bigEnd(1));
            m_fire[lev]->GetPhiField(m_phi_new[lev]);
            if (!m_feedback_diagnostic_file.empty()) {
                const auto step_stats = m_fire[lev]->GetLastPhiCorrectionStats();
                auto& accum = m_feedback_diagnostic_accum;
                accum.changed_cells += step_stats.changed_cells;
                accum.newly_burned_cells += step_stats.newly_burned_cells;
                accum.edge_newly_burned_cells += step_stats.edge_newly_burned_cells;
                accum.newly_tagged_nodes += step_stats.newly_tagged_nodes;
                accum.newly_burned_nodes += step_stats.newly_burned_nodes;
                accum.existing_nodes_activated += step_stats.existing_nodes_activated;
                accum.total_phi_decrease += step_stats.total_phi_decrease;
                accum.max_phi_decrease = std::max(accum.max_phi_decrease,
                                                 step_stats.max_phi_decrease);
                accum.new_node_fuel_frac_sum += step_stats.new_node_fuel_frac_sum;
            }
            if (!m_feedback_diagnostic_file.empty() &&
                m_istep[lev] % m_feedback_diagnostic_interval == 0) {
                auto stats = m_feedback_diagnostic_accum;
                m_feedback_diagnostic_accum = {};
                ParallelDescriptor::ReduceIntSum(stats.changed_cells);
                ParallelDescriptor::ReduceIntSum(stats.newly_burned_cells);
                ParallelDescriptor::ReduceIntSum(stats.edge_newly_burned_cells);
                ParallelDescriptor::ReduceIntSum(stats.newly_tagged_nodes);
                ParallelDescriptor::ReduceIntSum(stats.newly_burned_nodes);
                ParallelDescriptor::ReduceIntSum(stats.existing_nodes_activated);
                ParallelDescriptor::ReduceRealSum(stats.total_phi_decrease);
                ParallelDescriptor::ReduceRealMax(stats.max_phi_decrease);
                ParallelDescriptor::ReduceRealSum(stats.new_node_fuel_frac_sum);
                if (ParallelDescriptor::IOProcessor()) {
                    if (!m_feedback_diagnostic_stream.is_open()) {
                        m_feedback_diagnostic_stream.open(m_feedback_diagnostic_file);
                        if (!m_feedback_diagnostic_stream) {
                            Abort("Cannot open feedback diagnostic CSV");
                        }
                        m_feedback_diagnostic_stream
                            << "step,time,changed_cells,newly_burned_cells,"
                               "edge_newly_burned_cells,newly_tagged_nodes,"
                               "newly_burned_nodes,existing_nodes_activated,"
                               "total_phi_decrease,max_phi_decrease,"
                               "new_node_fuel_frac_sum\n";
                    }
                    m_feedback_diagnostic_stream
                        << m_istep[lev] << ',' << m_t_new[lev] << ','
                        << stats.changed_cells << ',' << stats.newly_burned_cells << ','
                        << stats.edge_newly_burned_cells << ','
                        << stats.newly_tagged_nodes << ','
                        << stats.newly_burned_nodes << ','
                        << stats.existing_nodes_activated << ','
                        << stats.total_phi_decrease << ','
                        << stats.max_phi_decrease << ','
                        << stats.new_node_fuel_frac_sum << '\n';
                    m_feedback_diagnostic_stream.flush();
                }
            }
        }
    }
    RecordArrivalTime(lev, time, dt);
}

void FireHierarchy::RecordArrivalTime(int lev, Real time, Real dt)
{
    if (!m_arrival_time[lev].ok()) return;
    RecordArrivalTimeKernel(m_arrival_time[lev], m_phi_old[lev],
                            m_phi_new[lev], time, dt);
}

// ============================================================================
// AdvanceLevel
// ============================================================================

void FireHierarchy::AdvanceLevel(int lev, amrex::Real time, amrex::Real dt)
{
    if (m_refinement_mode == RefinementMode::Static && finest_level > 0 &&
        m_fire_dt > 0.0) {
        Abort("Static multilevel fire requires fire.fire_dt=0 so each engine "
              "advances by its own AMR substep");
    }

    // Phase 1: any level with its own FireEngine advances its own physics
    // independently — no longer restricted to a single "the fire level".
    if (m_fire[lev]) {
        // Check for delayed ignition. m_ignited is shared across all levels
        // (not per-level), so the first level to cross m_ignition_time
        // ignites EVERY level's engine at once; subsequent levels' calls
        // within the same timestep then see m_ignited==true and skip this.
        if (!m_ignited && time >= m_ignition_time) {
            for (int l = 0; l <= finest_level; ++l) {
                if (m_fire[l]) {
                    IgniteLevel(l, time, false);
                }
            }
            m_ignited = true;
        }

        // Phase 4 (static multilevel plan): fill this level's ghost cells
        // from the coarser level before advancing, and push them into the
        // engine's own phi via the non-destructive sync (does not touch
        // narrow-band history — see SyncPhiFromHierarchy / KS-HISTORY-001).
        // Normal Advance() reads filled ghost phi in AddNewlyBurnedCells()
        // and in near-boundary gradient stencils. TagNarrowBand() is used
        // for ignition/restart, not for this per-step activation path.
        if (lev > 0) {
            // Two bugs were found and fixed here via AddressSanitizer
            // (2026-09-24) before this was safe to leave enabled:
            //
            // 1. FillPhi(lev, time, m_phi_new[lev]) internally passes
            //    m_phi_new[lev] as BOTH the destination and (as fmf[0]) one
            //    of FillPatchTwoLevels' own sources — aliasing destination
            //    and source. Fixed by filling into a separate MultiFab.
            //
            // 2. That separate MultiFab must NOT be a fresh local variable
            //    allocated every call: at fine levels AdvanceLevel runs
            //    every substep (thousands of times per run), and repeated
            //    allocate/free of a scratch MultiFab churns AMReX's Arena
            //    until it corrupts unrelated heap allocations. ASan caught
            //    the actual symptom as a heap-buffer-overflow inside a
            //    later, unrelated FillPlotVar() call — not at the
            //    allocation site — which is what made it hard to find by
            //    inspection alone. Fixed by using the persistent per-level
            //    m_phi_ghost_filled[lev] member (same lifecycle as
            //    m_phi_new/m_phi_old) instead of a local variable.
            FillPhi(lev, time, m_phi_ghost_filled[lev]);

            // Phase 4, narrowed scope (2026-09-24, see
            // docs/DEVELOPMENT_LOG_2026-09-24.md): switched from
            // SyncPhiFromHierarchy() (valid+ghost copy) to
            // FillGhostPhiFromHierarchy() (ghost-only, valid region
            // structurally unchanged) after the valid+ghost version caused
            // an unexplained fire_area regression under dynamic AMR (9.1M
            // baseline -> 2.35M m^2 over a 6h run, root cause not found).
            //
            // Validated (Test 5, static corridor, no runtime regrid):
            // ghost-fill ON vs OFF give bit-identical fire_area AND
            // narrow_band at t=1801.8s/3600s/5400s (before the fire's flank
            // reaches the corridor edge); they diverge starting at t=7200s,
            // right around when the flank should reach the edge — the
            // expected signature of ghost data actually mattering to
            // near-boundary nodes' own gradient/RK2 stencils, not a bug.
            // This does NOT validate the dynamic-AMR case (still open, see
            // dev log) — only static mode (fixed L0/L1, no runtime regrid,
            // hierarchy built before ignition) is currently trusted.
            m_fire[lev]->FillGhostPhiFromHierarchy(m_phi_ghost_filled[lev]);
        }

        // Advance fire model — sub-cycle internally if dt_cfl < fire_dt
        Real fire_dt = (m_fire_dt > 0.0) ? m_fire_dt : dt;
        m_fire[lev]->AdvanceInterval(fire_dt);

        // Copy updated phi back to AMR field
        m_fire[lev]->GetPhiField(m_phi_new[lev]);
    }

    // ------------------------------------------------------------------------
    // Accumulate fire scalars for time-averaged output — finest level only.
    //
    // Phase 1: kept scoped to finest_level exactly as before (NOT changed to
    // run per-level), because this block writes into m_cum_heat_flux_fire /
    // m_fire_ba_accum, which assume a single accumulator tied to one level's
    // BoxArray. Running it for every level would thrash m_fire_ba_accum
    // (different levels have different BoxArrays -> false-positive resets).
    // L1 accumulates its own heat and moisture. L0 separately
    // accumulates its own heat and moisture for cells outside L1 coverage.
    // ------------------------------------------------------------------------
    if (lev == finest_level && m_fire[lev]) {
            MFInfo host_info;
            host_info.SetArena(The_Managed_Arena());
            MultiFab hf(boxArray(lev), DistributionMap(lev), 1, 0, host_info);
            MultiFab mf(boxArray(lev), DistributionMap(lev), 1, 0, host_info);
            GetLevelHeatFlux(lev, hf);
            GetLevelMoistureFlux(lev, mf);

            // Reset L1 accumulators if BoxArray changed (after regrid)
            if (boxArray(lev) != m_fire_ba_accum) {
                m_cum_heat_flux_fire.define(boxArray(lev), DistributionMap(lev), 1, 0);
                m_cum_moist_flux_fire.define(boxArray(lev), DistributionMap(lev), 1, 0);
                m_cum_heat_flux_fire.setVal(0.0);
                m_cum_moist_flux_fire.setVal(0.0);
                m_fire_ba_accum = boxArray(lev);
                m_cum_time_fire = 0.0;
            }

            // Accumulate Σ(scalar × dt) at fire level (L1)
            MultiFab tmp(boxArray(lev), DistributionMap(lev), 1, 0);
            MultiFab::Copy(tmp, hf, 0, 0, 1, 0); tmp.mult(dt);
            MultiFab::Add(m_cum_heat_flux_fire, tmp, 0, 0, 1, 0);
            MultiFab::Copy(tmp, mf, 0, 0, 1, 0); tmp.mult(dt);
            MultiFab::Add(m_cum_moist_flux_fire, tmp, 0, 0, 1, 0);
            m_cum_time_fire += dt;


    }

    // In a composite solution L0 supplies the region outside L1. Accumulate
    // its own flux there instead of averaging L1 flux over covered coarse cells.
    if (lev == 0 && finest_level > 0 && m_fire[0]) {
        MFInfo host_info;
        host_info.SetArena(The_Managed_Arena());
        MultiFab hf(boxArray(0), DistributionMap(0), 1, 0, host_info);
        MultiFab mf(boxArray(0), DistributionMap(0), 1, 0, host_info);
        GetLevelHeatFlux(0, hf);
        GetLevelMoistureFlux(0, mf);
        hf.mult(dt);
        mf.mult(dt);
        MultiFab::Add(m_cum_heat_flux, hf, 0, 0, 1, 0);
        MultiFab::Add(m_cum_moist_flux, mf, 0, 0, 1, 0);
        m_cum_time += dt;
    }
}

// ============================================================================
// ComputeDt
// ============================================================================

amrex::Real FireHierarchy::ComputeDt(int lev)
{
    Real dt;
    if (m_dt_coarse > 0.0) {
        // User specified time.dt in inputs — use directly
        dt = m_dt_coarse;
    } else {
        // Fall back: CFL-based conservative estimate (max fire speed ~10 m/s)
        Real dx = Geom(lev).CellSize(0);
        dt = m_cfl * dx / 10.0;
    }

    // Limit by stop time
    if (m_t_new[lev] + dt > m_stop_time) {
        dt = m_stop_time - m_t_new[lev];
    }

    return std::max(dt, 1.0e-10);
}

// ============================================================================
// AverageDownPhi
//
// Not called as of Phase 1 (multilevel_static branch): every level now owns
// and advances its own phi independently (see timeStep/AdvanceLevel), so
// unconditionally overwriting a coarse level's phi with the fine level's
// average would defeat that. Kept for reuse once deliberate coarse/fine phi
// coupling is introduced (docs/LSFIRE_static_multilevel_plan.md, Phase 4).
// ============================================================================

void FireHierarchy::AverageDownPhi(int crse_lev)
{
    int fine_lev = crse_lev + 1;
    if (fine_lev > finest_level) return;

    average_down(m_phi_new[fine_lev], m_phi_new[crse_lev],
                 Geom(fine_lev), Geom(crse_lev),
                 0, 1, refRatio(crse_lev));
}

// Minimum restriction is defined only on covered coarse cells. Uncovered
// cells retain the coarse engine's phi in the candidate MultiFab.
void FireHierarchy::MinDownPhi(int crse_lev)
{
    const int fine_lev = crse_lev + 1;
    const IntVect ratio = refRatio(crse_lev);
    if (ratio[0] != 2 || ratio[1] != 2 || ratio[2] != 2) {
        Abort("static minimum feedback currently requires ref_ratio=2 2 2");
    }
    auto& coarse_blocks = m_phi_min_coarsened[crse_lev];
    auto& candidate = m_phi_min_candidate[crse_lev];
    const BoxArray coarse_fine_ba = amrex::coarsen(boxArray(fine_lev), ratio);
    if (!coarse_blocks.isDefined() || coarse_blocks.boxArray() != coarse_fine_ba) {
        coarse_blocks.define(coarse_fine_ba, DistributionMap(fine_lev), 1, 0);
    }
    if (!candidate.isDefined() || candidate.boxArray() != boxArray(crse_lev)) {
        candidate.define(boxArray(crse_lev), DistributionMap(crse_lev), 1, 0);
    }
    RestrictMinFeedbackKernel(coarse_blocks, m_phi_new[fine_lev],
                              m_static_min_feedback_min_burned,
                              m_static_min_feedback_sign_only);
    MultiFab::Copy(candidate, m_phi_new[crse_lev], 0, 0, 1, 0);
    candidate.ParallelCopy(coarse_blocks, 0, 0, 1, 0, 0);
}

// ============================================================================
// FillPhi (FillPatch for ghost cells)
// ============================================================================

void FireHierarchy::FillPhi(int lev, amrex::Real time, amrex::MultiFab& phi)
{
    // foextrap is handled by AMReX; no user ext_dir values are needed.
#ifdef AMREX_USE_GPU
    GpuBndryFuncFab<FirePhiBoundaryFill> bndry_func(FirePhiBoundaryFill{});
    using FirePhysBC = PhysBCFunct<GpuBndryFuncFab<FirePhiBoundaryFill>>;
#else
    CpuBndryFuncFab bndry_func(nullptr);
    using FirePhysBC = PhysBCFunct<CpuBndryFuncFab>;
#endif

    if (lev == 0) {
        // Level 0: single-level fill (periodic/extrapolation BCs)
        Vector<MultiFab*> smf = {&m_phi_new[lev]};
        Vector<Real> stime = {m_t_new[lev]};

        FirePhysBC physbc(Geom(lev), m_bcrec, bndry_func);
        FillPatchSingleLevel(phi, time, smf, stime,
                              0, 0, 1, Geom(lev),
                              physbc, 0);
    } else {
        // Fine levels: two-level fill (interpolate from coarse)
        Vector<MultiFab*> cmf = {&m_phi_old[lev-1], &m_phi_new[lev-1]};
        Vector<Real> ctime = {m_t_old[lev-1], m_t_new[lev-1]};

        Vector<MultiFab*> fmf = {&m_phi_new[lev]};
        Vector<Real> ftime = {m_t_new[lev]};

        FirePhysBC cphysbc(Geom(lev-1), m_bcrec, bndry_func);
        FirePhysBC fphysbc(Geom(lev), m_bcrec, bndry_func);

        Interpolater* mapper = m_phi_bilinear_interp
            ? static_cast<Interpolater*>(&cell_bilinear_interp)
            : static_cast<Interpolater*>(&cell_cons_interp);
        FillPatchTwoLevels(phi, time, cmf, ctime, fmf, ftime,
                            0, 0, 1,
                            Geom(lev-1), Geom(lev),
                            cphysbc, 0,
                            fphysbc, 0,
                            refRatio(lev-1),
                            mapper,
                            m_bcrec, 0);
    }
}

// ============================================================================
// MakeNewLevelFromScratch (initialization)
// ============================================================================

void FireHierarchy::MakeNewLevelFromScratch(int lev, amrex::Real time,
                                              const amrex::BoxArray& ba,
                                              const amrex::DistributionMapping& dm)
{
    MFInfo phi_info;
    phi_info.SetArena(The_Managed_Arena());
    m_phi_new[lev].define(ba, dm, 1, m_phi_nghost, phi_info);
    m_phi_old[lev].define(ba, dm, 1, m_phi_nghost, phi_info);
    m_phi_ghost_filled[lev].define(ba, dm, 1, m_phi_nghost, phi_info);
    m_arrival_time[lev].define(ba, dm, 1, 0);
    m_arrival_time[lev].setVal(-1.0);

    m_t_new[lev] = time;
    m_t_old[lev] = time - 1.0e10;

    // Initialize phi = 1.0 everywhere (unburned)
    m_phi_new[lev].setVal(1.0);
    m_phi_old[lev].setVal(1.0);
}

// ============================================================================
// MakeNewLevelFromCoarse (regridding: new fine level)
// ============================================================================

void FireHierarchy::MakeNewLevelFromCoarse(int lev, amrex::Real time,
                                             const amrex::BoxArray& ba,
                                             const amrex::DistributionMapping& dm)
{
    MFInfo phi_info;
    phi_info.SetArena(The_Managed_Arena());
    m_phi_new[lev].define(ba, dm, 1, m_phi_nghost, phi_info);
    m_phi_old[lev].define(ba, dm, 1, m_phi_nghost, phi_info);
    m_phi_ghost_filled[lev].define(ba, dm, 1, m_phi_nghost, phi_info);
    m_arrival_time[lev].define(ba, dm, 1, 0);
    m_arrival_time[lev].setVal(-1.0);

    m_t_new[lev] = time;
    m_t_old[lev] = time;

    // Fill new level by interpolating from coarse
    FillPhi(lev, time, m_phi_new[lev]);
    MultiFab::Copy(m_phi_old[lev], m_phi_new[lev], 0, 0, 1, m_phi_nghost);

    // This new level's own fire engine is created lazily in timeStep's
    // regrid-recovery block (right after regrid() returns), not here.
    //
    // Phase 1: coarser levels' fire engines are NO LONGER destroyed when a
    // finer level is created. Every level keeps running its own FireEngine
    // independently (see AdvanceLevel) — previously this loop reset
    // m_fire[l] for all l < lev, which enforced the old "only finest_level
    // runs physics" architecture.
}

// ============================================================================
// RemakeLevel (regridding: existing level gets new grids)
// ============================================================================

void FireHierarchy::RemakeLevel(int lev, amrex::Real time,
                                  const amrex::BoxArray& ba,
                                  const amrex::DistributionMapping& dm)
{
    // Create temporary MultiFabs with new grids
    MFInfo phi_info;
    phi_info.SetArena(The_Managed_Arena());
    MultiFab new_phi(ba, dm, 1, m_phi_nghost, phi_info);
    MultiFab new_phi_old(ba, dm, 1, m_phi_nghost, phi_info);

    // Fill from old data + coarse interpolation
    FillPhi(lev, time, new_phi);
    MultiFab::Copy(new_phi_old, new_phi, 0, 0, 1, m_phi_nghost);

    // Swap
    std::swap(m_phi_new[lev], new_phi);
    std::swap(m_phi_old[lev], new_phi_old);
    MultiFab new_arrival(ba, dm, 1, 0);
    new_arrival.setVal(-1.0);
    new_arrival.ParallelCopy(m_arrival_time[lev], 0, 0, 1);
    std::swap(m_arrival_time[lev], new_arrival);

    // Scratch ghost-fill buffer just needs to match the new grid; no data
    // to preserve across a regrid (it's rewritten every AdvanceLevel call).
    m_phi_ghost_filled[lev].define(ba, dm, 1, m_phi_nghost, phi_info);

    m_t_new[lev] = time;
    m_t_old[lev] = time;

    // Regrid fire engine in-place: redistribute MultiFabs and migrate
    // narrow-band nodes to the correct MPI rank (Option 2).
    // This preserves all per-node state without destroying/recreating the engine.
    if (m_fire[lev]) {
        m_fire[lev]->Regrid(ba, dm);
    }
}

// ============================================================================
// ClearLevel
// ============================================================================

void FireHierarchy::ClearLevel(int lev)
{
    m_phi_new[lev].clear();
    m_phi_old[lev].clear();
    m_phi_ghost_filled[lev].clear();
    m_arrival_time[lev].clear();
    m_fire[lev].reset();
}

// ============================================================================
// ErrorEst (cell tagging for refinement)
// ============================================================================

void FireHierarchy::ErrorEst(int lev, amrex::TagBoxArray& tags,
                               amrex::Real /*time*/, int /*ngrow*/)
{
    // -------------------------------------------------------------------------
    // Phase 2/3 (static multilevel plan): fixed rectangular corridor,
    // independent of fire position. Tagged once (see timeStep, which skips
    // periodic regridding in this mode) and never re-tagged from the front.
    // Deliberately NOT combined with the dynamic narrow-band logic below —
    // see docs/LSFIRE_static_multilevel_plan.md Phase 2.
    // -------------------------------------------------------------------------
    if (m_refinement_mode == RefinementMode::Static) {
        if (!m_static_refine_mask.empty()) {
            if (lev != 0) Abort("Static refinement mask currently supports L0 tagging only");
            Gpu::DeviceVector<unsigned char> device_mask(m_static_refine_mask.size());
            Gpu::copy(Gpu::hostToDevice, m_static_refine_mask.begin(),
                      m_static_refine_mask.end(), device_mask.begin());
            const unsigned char* mask = device_mask.dataPtr();
            const int nx = m_static_mask_nx;
            const auto domain = Geom(0).Domain();
            const int ilo = domain.smallEnd(0);
            const int jlo = domain.smallEnd(1);
            for (MFIter mfi(tags); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.tilebox();
                auto tag_arr = tags.array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    if (mask[(j-jlo)*nx + (i-ilo)] != 0) tag_arr(i,j,k) = TagBox::SET;
                });
            }
            Gpu::streamSynchronize();
            return;
        }
        if (!m_static_polyline_segments.empty()) {
            Gpu::DeviceVector<Real> device_segments(m_static_polyline_segments.size());
            Gpu::copy(Gpu::hostToDevice, m_static_polyline_segments.begin(),
                      m_static_polyline_segments.end(), device_segments.begin());
            const Real* segments = device_segments.dataPtr();
            const int nsegments = static_cast<int>(m_static_polyline_segments.size() / 4);
            const Real half_width_sq = 0.25 * m_static_corridor_width * m_static_corridor_width;
            const Real* problo = Geom(lev).ProbLo();
            const Real dx_lev = Geom(lev).CellSize(0);
            const Real dy_lev = Geom(lev).CellSize(1);
            for (MFIter mfi(tags); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.tilebox();
                auto tag_arr = tags.array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    const Real x = problo[0] + (i + 0.5) * dx_lev;
                    const Real y = problo[1] + (j + 0.5) * dy_lev;
                    for (int n = 0; n < nsegments; ++n) {
                        const Real x0 = segments[4*n];
                        const Real y0_seg = segments[4*n+1];
                        const Real vx = segments[4*n+2] - x0;
                        const Real vy = segments[4*n+3] - y0_seg;
                        const Real t = amrex::max(Real(0.0), amrex::min(Real(1.0),
                            ((x-x0)*vx + (y-y0_seg)*vy) / (vx*vx + vy*vy)));
                        const Real rx = x - (x0 + t*vx);
                        const Real ry = y - (y0_seg + t*vy);
                        if (rx*rx + ry*ry <= half_width_sq) {
                            tag_arr(i,j,k) = TagBox::SET;
                            break;
                        }
                    }
                });
            }
            Gpu::streamSynchronize();
            return;
        }
        const Real y0 = m_static_corridor_y0;
        const Real half_width = 0.5 * m_static_corridor_width;
        const Real amplitude = m_static_corridor_amplitude;
        const Real wavelength = m_static_corridor_wavelength;
        const Real* problo = Geom(lev).ProbLo();
        const Real dx_lev = Geom(lev).CellSize(0);
        const Real dy_lev = Geom(lev).CellSize(1);

        for (MFIter mfi(tags); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            auto tag_arr = tags.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const Real x_cell = problo[0] + (i + 0.5) * dx_lev;
                const Real y_cell = problo[1] + (j + 0.5) * dy_lev;
                const Real center = amplitude == 0.0 ? y0
                    : y0 + amplitude * std::sin(2.0 * 3.14159265358979323846
                                                   * (x_cell - problo[0]) / wavelength);
                if (y_cell >= center - half_width && y_cell <= center + half_width) {
                    tag_arr(i, j, k) = TagBox::SET;
                }
            });
        }
        return;
    }

    // -------------------------------------------------------------------------
    // Dynamic mode (default, unchanged from pre-Phase-2 behavior):
    // Option A: Tag based on narrow-band node positions only.
    //
    // - If fire engine exists on the fine level: iterate narrow-band nodes,
    //   convert fine-level (ix,iy) to coarse-level cell indices, and tag
    //   those coarse cells + 1-cell buffer.
    // - If no fire engine yet (before/at ignition): tag ignition point + buffer.
    // -------------------------------------------------------------------------

    const Real dx_lev = Geom(lev).CellSize(0);
    const Real* prob_lo = Geom(lev).ProbLo();
    const Box& domain = Geom(lev).Domain();

    int fire_lev = FireLevel();

    // Check if fire engine exists on a finer level
    bool have_fire = (fire_lev > lev) && m_fire[fire_lev];

    if (have_fire) {
        // -----------------------------------------------------------------
        // Build a tag mask from narrow-band node positions on the fire level.
        // Convert fine-level (ix,iy) to coarse-level cell indices and tag
        // those coarse cells + 1-cell buffer.
        //
        // MPI: Narrow-band nodes are distributed across ranks on the fine
        // level, but the corresponding coarse cells may be on different ranks.
        // We must gather ALL narrow-band positions across all ranks so each
        // rank can tag its own coarse cells correctly.
        // -----------------------------------------------------------------
        const int rr = refRatio(lev)[0]; // refinement ratio (e.g., 2 or 4)

        // Step 1: Each rank gets its local narrow-band cells and converts
        //         to coarse indices (unique set to avoid duplicates)
        auto local_nb = m_fire[fire_lev]->GetNarrowBandCells();

        // Convert to coarse indices (stored as flat array: [ci0,cj0,ci1,cj1,...])
        // Use a set to deduplicate (many fine cells map to the same coarse cell)
        std::set<std::pair<int,int>> local_coarse_set;
        for (const auto& [fix, fiy] : local_nb) {
            int ci = fix / rr;
            int cj = fiy / rr;
            // Add the coarse cell + 1-cell buffer
            for (int jj = cj - 1; jj <= cj + 1; ++jj) {
                for (int ii = ci - 1; ii <= ci + 1; ++ii) {
                    if (ii >= domain.smallEnd(0) && ii <= domain.bigEnd(0) &&
                        jj >= domain.smallEnd(1) && jj <= domain.bigEnd(1)) {
                        local_coarse_set.insert({ii, jj});
                    }
                }
            }
        }

        // Flatten to vector for MPI communication
        std::vector<int> local_data;
        local_data.reserve(local_coarse_set.size() * 2);
        for (const auto& [ci, cj] : local_coarse_set) {
            local_data.push_back(ci);
            local_data.push_back(cj);
        }

        // Step 2: AllGather coarse cell indices across all MPI ranks
        int local_count = static_cast<int>(local_data.size());
        int nprocs = ParallelDescriptor::NProcs();
        std::vector<int> all_counts(nprocs);
        MPI_Allgather(&local_count, 1, MPI_INT,
                       all_counts.data(), 1, MPI_INT,
                       ParallelDescriptor::Communicator());

        // Compute displacements for Allgatherv
        std::vector<int> displs(nprocs, 0);
        int total_count = all_counts[0];
        for (int p = 1; p < nprocs; ++p) {
            displs[p] = displs[p-1] + all_counts[p-1];
            total_count += all_counts[p];
        }

        std::vector<int> all_data(total_count);
        MPI_Allgatherv(local_data.data(), local_count, MPI_INT,
                        all_data.data(), all_counts.data(), displs.data(),
                        MPI_INT, ParallelDescriptor::Communicator());

        // Step 3: Each rank tags its own coarse cells from the gathered list
        for (MFIter mfi(tags); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            auto tag_arr = tags.array(mfi);

            // For each gathered coarse cell, check if it falls in this box
            for (int idx = 0; idx < total_count; idx += 2) {
                int ci = all_data[idx];
                int cj = all_data[idx + 1];
                if (bx.contains(IntVect(amrex::Dim3{AMREX_D_DECL(ci, cj, 0})))) {
                    for (int kk = bx.smallEnd(2); kk <= bx.bigEnd(2); ++kk) {
                        tag_arr(ci, cj, kk) = TagBox::SET;
                    }
                }
            }
        }

    } else {
        // -----------------------------------------------------------------
        // No fire engine yet: tag ignition point + buffer
        // -----------------------------------------------------------------
        int ign_i = static_cast<int>((m_ignition_x - prob_lo[0]) / dx_lev);
        int ign_j = static_cast<int>((m_ignition_y - prob_lo[1]) / dx_lev);
        // Buffer around ignition: enough for narrow band to form (band_thickness + margin)
        int ign_buffer = static_cast<int>(m_fire_config.band_thickness) + 2;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(m_phi_new[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            auto tag_arr = tags.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                if (std::abs(i - ign_i) <= ign_buffer &&
                    std::abs(j - ign_j) <= ign_buffer) {
                    tag_arr(i, j, k) = TagBox::SET;
                }
            });
        }
    }
}

amrex::iMultiFab FireHierarchy::GetCompositeValidMask(int lev) const
{
    AMREX_ALWAYS_ASSERT(lev >= 0 && lev <= finest_level);
    if (lev < finest_level) {
        return makeFineMask(boxArray(lev), DistributionMap(lev),
                            boxArray(lev + 1), refRatio(lev), 1, 0);
    }
    iMultiFab mask(boxArray(lev), DistributionMap(lev), 1, 0);
    mask.setVal(1);
    return mask;
}

void FireHierarchy::GetLevelHeatFlux(int lev, MultiFab& dst) const
{
    AMREX_ALWAYS_ASSERT(lev >= 0 && lev <= finest_level && m_fire[lev]);
    AMREX_ALWAYS_ASSERT(dst.boxArray() == boxArray(lev));
    AMREX_ALWAYS_ASSERT(dst.DistributionMap() == DistributionMap(lev));
    m_fire[lev]->GetHeatFlux(dst);
}

void FireHierarchy::GetLevelMoistureFlux(int lev, MultiFab& dst) const
{
    AMREX_ALWAYS_ASSERT(lev >= 0 && lev <= finest_level && m_fire[lev]);
    AMREX_ALWAYS_ASSERT(dst.boxArray() == boxArray(lev));
    AMREX_ALWAYS_ASSERT(dst.DistributionMap() == DistributionMap(lev));
    m_fire[lev]->GetMoistureFlux(dst);
}

// Count one horizontal fire plane per AMR level. Covered coarse cells are
// excluded, so this agrees with fine-over-coarse plotfile composition.
amrex::Real FireHierarchy::GetCompositeBurnedArea() const
{
    Real area = 0.0;
    for (int lev = 0; lev <= finest_level; ++lev) {
        if (!m_phi_new[lev].ok()) continue;
        iMultiFab mask = GetCompositeValidMask(lev);
        MultiFab burned(boxArray(lev), DistributionMap(lev), 1, 0);
        for (MFIter mfi(burned, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto dst = burned.array(mfi);
            auto phi = m_phi_new[lev].const_array(mfi);
            auto valid = mask.const_array(mfi);
            const Box bx = mfi.tilebox();
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                dst(i,j,k) = (k == 0 && valid(i,j,k) != 0 && phi(i,j,k) <= 0.0)
                    ? Real(1.0) : Real(0.0);
            });
        }
        const auto dx = Geom(lev).CellSizeArray();
        area += burned.sum(0) * dx[0] * dx[1];
    }
    return area;
}

// ============================================================================
// WritePlotFile
// ============================================================================
// Each level writes its own solution. fire_composite_valid is one on cells
// selected by fine-over-coarse precedence and zero on covered coarse cells.
// A composite reader selects cells with that mask, without summing overlap.
void FireHierarchy::WritePlotFile(int step)
{
    WritePlotFile(m_output_dir + "/fire_amr_" + std::to_string(step));
}

void FireHierarchy::WritePlotFile(const std::string& filename)
{
    static const std::array<std::string, 3> fire_scalar_keys = {
        "fire_heat_flux", "fire_moisture_flux", "fire_fuel_frac"
    };
    static const std::array<std::string, 3> fire_front_keys = {
        "fire_vel_u", "fire_vel_v", "fire_tagged"
    };
    Vector<std::string> varnames = {"fire_phi"};
    for (const auto& key : fire_scalar_keys) varnames.push_back(key);
    for (const auto& key : fire_front_keys) varnames.push_back(key);
    Vector<std::string> extra_names;
    GetExtraPlotVarNames(extra_names);
    for (const auto& key : extra_names) varnames.push_back(key);
    const int mask_comp = static_cast<int>(varnames.size());
    varnames.push_back("fire_composite_valid");
    const int arrival_comp = static_cast<int>(varnames.size());
    varnames.push_back("fire_arrival_time");

    Vector<MultiFab> output_mfs(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
        auto& out = output_mfs[lev];
        out.define(boxArray(lev), DistributionMap(lev), varnames.size(), 0);
        out.setVal(0.0);
        MultiFab::Copy(out, m_phi_new[lev], 0, 0, 1, 0);
        MultiFab::Copy(out, m_arrival_time[lev], 0, arrival_comp, 1, 0);

        iMultiFab valid = GetCompositeValidMask(lev);
        FillValidMaskKernel(out, valid, mask_comp);

        if (!m_fire[lev]) continue;
        for (int n = 0; n < 3; ++n) {
            MFInfo host_info;
            host_info.SetArena(The_Managed_Arena());
            MultiFab tmp(boxArray(lev), DistributionMap(lev), 1, 0, host_info);
            m_fire[lev]->FillPlotVar(fire_scalar_keys[n], tmp);
            MultiFab::Copy(out, tmp, 0, 1 + n, 1, 0);
        }
        for (int n = 0; n < 3; ++n) {
            MFInfo host_info;
            host_info.SetArena(The_Managed_Arena());
            MultiFab tmp(boxArray(lev), DistributionMap(lev), 1, 0, host_info);
            tmp.setVal(0.0);
            m_fire[lev]->FillPlotVar(fire_front_keys[n], tmp);
            MultiFab::Copy(out, tmp, 0, 4 + n, 1, 0);
        }
    }

    // Keep the existing interval-mean convention for fluxes. Both levels
    // now accumulate their own fire, and coarse cells under L1 are masked.
    if (m_cum_time > 0.0) {
        MultiFab::Copy(output_mfs[0], m_cum_heat_flux, 0, 1, 1, 0);
        MultiFab::Copy(output_mfs[0], m_cum_moist_flux, 0, 2, 1, 0);
        output_mfs[0].mult(1.0 / m_cum_time, 1, 2);
    }
    const int fire_lev = finest_level;
    if (m_fire[fire_lev] && m_cum_time_fire > 0.0 &&
        m_cum_heat_flux_fire.ok() && m_fire_ba_accum == boxArray(fire_lev)) {
        MultiFab::Copy(output_mfs[fire_lev], m_cum_heat_flux_fire, 0, 1, 1, 0);
        MultiFab::Copy(output_mfs[fire_lev], m_cum_moist_flux_fire, 0, 2, 1, 0);
        output_mfs[fire_lev].mult(1.0 / m_cum_time_fire, 1, 2);
    }
    // Subclass extras are written on every level (metgrid wind is level-specific).
    for (int lev = 0; lev <= finest_level; ++lev) {
        for (int n = 0; n < static_cast<int>(extra_names.size()); ++n) {
            MultiFab tmp(boxArray(lev), DistributionMap(lev), 1, 0);
            tmp.setVal(0.0);
            FillExtraPlotVar(extra_names[n], tmp);
            MultiFab::Copy(output_mfs[lev], tmp, 0, 7 + n, 1, 0);
        }
    }

    // -------------------------------------------------------------------------
    // Write multi-level plotfile
    // -------------------------------------------------------------------------
    Vector<const MultiFab*> output_ptrs(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) output_ptrs[lev] = &output_mfs[lev];

    Vector<Geometry> geoms(finest_level + 1);
    Vector<int> level_steps(finest_level + 1);
    Vector<IntVect> ref_ratios(finest_level);
    for (int lev = 0; lev <= finest_level; ++lev) {
        geoms[lev] = Geom(lev);
        level_steps[lev] = m_istep[lev];
    }
    for (int lev = 0; lev < finest_level; ++lev) {
        ref_ratios[lev] = refRatio(lev);
    }

    WriteMultiLevelPlotfile(filename, finest_level + 1,
                             output_ptrs, varnames,
                             geoms, m_t_new[0],
                             level_steps, ref_ratios);

    // Reset accumulators for the next output interval
    m_cum_heat_flux.setVal(0.0);
    m_cum_moist_flux.setVal(0.0);
    m_cum_time = 0.0;

    // Reset L1 accumulators (BoxArray stays valid; just zero the values)
    if (m_cum_heat_flux_fire.ok()) {
        m_cum_heat_flux_fire.setVal(0.0);
        m_cum_moist_flux_fire.setVal(0.0);
    }
    m_cum_time_fire = 0.0;
}
