#include "LevelSetSolver.H"
#include <AMReX_MFIter.H>
#include <AMReX_Gpu.H>
#include <cmath>
#include <algorithm>

using namespace amrex;

void Rk2Integrate(GpuNarrowBand& cells,
                  MultiFab& /*phi*/,
                  Real dt,
                  int istep)
{
    int N = cells.GetNumNodes();
    if (N == 0) return;

    cells.CopyToDevice();
    FireNodeData* nodes_d = cells.GetDevicePtr();

    if (istep == 1) {
        // Predictor: φ* = φⁿ - dt*(u·∇φ)
        amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept {
            FireNodeData& nd = nodes_d[i];
            Real adv = nd.ux * nd.dphidx_limited + nd.uy * nd.dphidy_limited;
            Real phi_new = nd.phip_old - dt * adv;
            if (amrex::isnan(phi_new)) phi_new = Real(1.0);
            phi_new = amrex::min(amrex::max(phi_new, Real(-100.0)), Real(100.0));
            nd.phip_rk1 = phi_new;
            nd.phi      = phi_new;
        });
    } else {
        // Corrector: φⁿ⁺¹ = 0.5*(φⁿ + φ* - dt*(u·∇φ*))
        amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept {
            FireNodeData& nd = nodes_d[i];
            Real adv = nd.ux * nd.dphidx_limited + nd.uy * nd.dphidy_limited;
            Real phi_new = Real(0.5) * (nd.phip_old + (nd.phip_rk1 - dt * adv));
            if (amrex::isnan(phi_new)) phi_new = Real(1.0);
            phi_new = amrex::min(amrex::max(phi_new, Real(-100.0)), Real(100.0));
            nd.phi = phi_new;
        });
    }

    cells.CopyFromDevice();
}
