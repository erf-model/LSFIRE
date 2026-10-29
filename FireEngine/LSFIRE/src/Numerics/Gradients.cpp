#include "Gradients.H"
#include <AMReX_MFIter.H>
#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>
#include <algorithm>

using namespace amrex;

namespace {
// Extinction masking: a dead burned neighbour carries no information to the receiver; it is read as unburned (+1).
// (State-constraint boundary; see docs/extinction_masking.md.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real PhiEff (Array4<Real const> const& p, Array4<Real const> const& d, bool has_dead, int i, int j) noexcept
{
    return (has_dead && d(i, j, 0) > Real(0.5)) ? Real(1.0) : p(i, j, 0);
}
}

void CalcNormalVectors(GpuNarrowBand& cells,
                       const MultiFab& phi,
                       Real half_rcellsize,
                       int istep,
                       const MultiFab* dead)
{
    constexpr Real EPSILON = 1.0e-30;
    constexpr Real BIG     = 3.0e4;

    int N = cells.GetNumNodes();
    if (N == 0) return;

    cells.CopyToDevice();
    FireNodeData* nodes_d = cells.GetDevicePtr();

    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box valid = mfi.validbox();
        const Box ghost = mfi.growntilebox();
        const auto phi_arr = phi.const_array(mfi);
        const bool has_dead = (dead != nullptr);
        const auto dead_arr = has_dead ? dead->const_array(mfi) : phi_arr;

        amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept {
            FireNodeData& nd = nodes_d[i];
            int IX = nd.ix, IY = nd.iy;

            if (!valid.contains(IntVect(amrex::Dim3{IX,   IY,   0}))) return;
            if (!ghost.contains(IntVect(amrex::Dim3{IX-1, IY,   0}))) return;
            if (!ghost.contains(IntVect(amrex::Dim3{IX+1, IY,   0}))) return;
            if (!ghost.contains(IntVect(amrex::Dim3{IX,   IY-1, 0}))) return;
            if (!ghost.contains(IntVect(amrex::Dim3{IX,   IY+1, 0}))) return;

            if (istep == 1) nd.phip_old = phi_arr(IX, IY, 0);

            Real DPHIDX = amrex::max(amrex::min(
                half_rcellsize * (PhiEff(phi_arr, dead_arr, has_dead, IX+1, IY) - PhiEff(phi_arr, dead_arr, has_dead, IX-1, IY)), BIG), -BIG);
            Real DPHIDY = amrex::max(amrex::min(
                half_rcellsize * (PhiEff(phi_arr, dead_arr, has_dead, IX, IY+1) - PhiEff(phi_arr, dead_arr, has_dead, IX, IY-1)), BIG), -BIG);

            Real mag_grad    = std::sqrt(DPHIDX * DPHIDX + DPHIDY * DPHIDY);
            Real rmag        = Real(1.0) / amrex::max(mag_grad, EPSILON);

            nd.normx   = rmag * DPHIDX;
            nd.normy   = rmag * DPHIDY;
            nd.dphidx  = DPHIDX;
            nd.dphidy  = DPHIDY;
            nd.gradmag = mag_grad;
        });
    }

    cells.CopyFromDevice();
}

void CflAndFluxLimiter(GpuNarrowBand& cells,
                       const MultiFab& phi,
                       Real rcellsize,
                       Real& dt,
                       int istep,
                       Real target_cfl,
                       const MultiFab* dead)
{
    constexpr Real EPSILON = 1.0e-30;

    int N = cells.GetNumNodes();

    // CPU: compute UMAX from host node data.
    // ReduceRealMax is an MPI collective — must be called on ALL ranks
    // regardless of whether this rank has any narrow-band nodes.
    Real UMAX = 0.0;
    if (istep == 1) {
        for (int i = 0; i < N; i++) {
            const FireNodeData& nd = cells[i];
            if (!nd.burned) {
                Real cond = amrex::max(amrex::Math::abs(nd.ux), amrex::Math::abs(nd.uy));
                UMAX = amrex::max(UMAX, cond);
            }
        }
        ParallelDescriptor::ReduceRealMax(UMAX);  // collective: all ranks participate
        if (UMAX > 0.0) {
            Real cellsize = Real(1.0) / rcellsize;
            dt = std::min(dt, target_cfl * cellsize / UMAX);
        }
    }

    if (N == 0) return;

    // GPU: init limited gradients, then MUSCL-Superbee per FAB
    cells.CopyToDevice();
    FireNodeData* nodes_d = cells.GetDevicePtr();

    // Pass 1: initialize all nodes' limited gradients to raw gradients
    amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept {
        nodes_d[i].dphidx_limited = nodes_d[i].dphidx;
        nodes_d[i].dphidy_limited = nodes_d[i].dphidy;
    });

    // Pass 2: overwrite with MUSCL-Superbee for nodes inside each FAB
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box valid = mfi.validbox();
        const Box ghost = mfi.growntilebox();
        const auto phi_arr = phi.const_array(mfi);
        const bool has_dead = (dead != nullptr);
        const auto dead_arr = has_dead ? dead->const_array(mfi) : phi_arr;

        amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept {
            FireNodeData& nd = nodes_d[i];
            int IX = nd.ix, IY = nd.iy;

            if (!valid.contains(IntVect(amrex::Dim3{IX,   IY,   0}))) return;
            if (!ghost.contains(IntVect(amrex::Dim3{IX-2, IY,   0}))) return;
            if (!ghost.contains(IntVect(amrex::Dim3{IX+2, IY,   0}))) return;
            if (!ghost.contains(IntVect(amrex::Dim3{IX,   IY-2, 0}))) return;
            if (!ghost.contains(IntVect(amrex::Dim3{IX,   IY+2, 0}))) return;

            Real phi_c   = phi_arr(IX,   IY,   0);
            Real phi_xp1 = PhiEff(phi_arr, dead_arr, has_dead, IX+1, IY);
            Real phi_xm1 = PhiEff(phi_arr, dead_arr, has_dead, IX-1, IY);
            Real phi_xp2 = PhiEff(phi_arr, dead_arr, has_dead, IX+2, IY);
            Real phi_xm2 = PhiEff(phi_arr, dead_arr, has_dead, IX-2, IY);
            Real phi_yp1 = PhiEff(phi_arr, dead_arr, has_dead, IX, IY+1);
            Real phi_ym1 = PhiEff(phi_arr, dead_arr, has_dead, IX, IY-1);
            Real phi_yp2 = PhiEff(phi_arr, dead_arr, has_dead, IX, IY+2);
            Real phi_ym2 = PhiEff(phi_arr, dead_arr, has_dead, IX, IY-2);

            // X-direction MUSCL-Superbee
            Real PHIEAST = phi_c, PHIWEST = phi_xm1;
            if (nd.ux >= 0.0) {
                Real DU_e = phi_c - phi_xm1,   DL_e = phi_xp1 - phi_c;
                if (amrex::Math::abs(DL_e) > EPSILON)
                    PHIEAST = phi_c + HalfSuperbee(DU_e / DL_e) * DL_e;
                Real DU_w = phi_xm1 - phi_xm2, DL_w = phi_c - phi_xm1;
                if (amrex::Math::abs(DL_w) > EPSILON)
                    PHIWEST = phi_xm1 + HalfSuperbee(DU_w / DL_w) * DL_w;
            } else {
                PHIEAST = phi_xp1; PHIWEST = phi_c;
                Real DU_e = phi_xp1 - phi_xp2, DL_e = phi_c - phi_xp1;
                if (amrex::Math::abs(DL_e) > EPSILON)
                    PHIEAST = phi_xp1 + HalfSuperbee(DU_e / DL_e) * DL_e;
                Real DU_w = phi_c - phi_xp1,   DL_w = phi_xm1 - phi_c;
                if (amrex::Math::abs(DL_w) > EPSILON)
                    PHIWEST = phi_c + HalfSuperbee(DU_w / DL_w) * DL_w;
            }
            nd.dphidx_limited = (PHIEAST - PHIWEST) * rcellsize;

            // Y-direction MUSCL-Superbee
            Real PHINORTH = phi_c, PHISOUTH = phi_ym1;
            if (nd.uy >= 0.0) {
                Real DU_n = phi_c - phi_ym1,   DL_n = phi_yp1 - phi_c;
                if (amrex::Math::abs(DL_n) > EPSILON)
                    PHINORTH = phi_c + HalfSuperbee(DU_n / DL_n) * DL_n;
                Real DU_s = phi_ym1 - phi_ym2, DL_s = phi_c - phi_ym1;
                if (amrex::Math::abs(DL_s) > EPSILON)
                    PHISOUTH = phi_ym1 + HalfSuperbee(DU_s / DL_s) * DL_s;
            } else {
                PHINORTH = phi_yp1; PHISOUTH = phi_c;
                Real DU_n = phi_yp1 - phi_yp2, DL_n = phi_c - phi_yp1;
                if (amrex::Math::abs(DL_n) > EPSILON)
                    PHINORTH = phi_yp1 + HalfSuperbee(DU_n / DL_n) * DL_n;
                Real DU_s = phi_c - phi_yp1,   DL_s = phi_ym1 - phi_c;
                if (amrex::Math::abs(DL_s) > EPSILON)
                    PHISOUTH = phi_c + HalfSuperbee(DU_s / DL_s) * DL_s;
            }
            nd.dphidy_limited = (PHINORTH - PHISOUTH) * rcellsize;

            // Extinction: a dead cell does not evolve; next to dead cells phi may not rise (no retreat)
            if (has_dead) {
                if (dead_arr(IX, IY, 0) > Real(0.5)) {
                    nd.dphidx_limited = Real(0.0); nd.dphidy_limited = Real(0.0);
                } else {
                    bool near_dead = false;
                    for (int k = 1; k <= 2; ++k) {
                        near_dead = near_dead || dead_arr(IX+k, IY, 0) > Real(0.5) || dead_arr(IX-k, IY, 0) > Real(0.5)
                                              || dead_arr(IX, IY+k, 0) > Real(0.5) || dead_arr(IX, IY-k, 0) > Real(0.5);
                    }
                    if (near_dead && nd.ux * nd.dphidx_limited + nd.uy * nd.dphidy_limited < Real(0.0)) {
                        nd.dphidx_limited = Real(0.0); nd.dphidy_limited = Real(0.0);
                    }
                }
            }
        });
    }

    cells.CopyFromDevice();
}
