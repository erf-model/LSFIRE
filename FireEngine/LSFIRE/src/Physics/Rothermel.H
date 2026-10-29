#ifndef ELMFIRE_ROTHERMEL_H
#define ELMFIRE_ROTHERMEL_H

/**
 * Rothermel surface fire spread rate model (Rothermel 1972).
 *
 * Implementation is a template so the same source compiles for both CPU
 * (FireNode&) and GPU (FireNodeData&) without code duplication.
 * The #ifndef __CUDA_ARCH__ guard strips out amrex::Print() debug sections
 * on GPU device, where I/O is not available.
 *
 * Reference: Rothermel 1972, USDA FS Research Paper INT-115.
 */

#include "FuelModels.H"
#include <AMReX_Algorithm.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
#ifndef __CUDA_ARCH__
#include <AMReX_Print.H>
#endif

template <typename NodeT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void SurfaceSpreadRate(NodeT& node, const FuelModelTable& table)
{
    using amrex::Real;
    using amrex::max;
    using amrex::min;

    constexpr Real BTUPFT2MIN_TO_KWPM2 = Real(1.055) / (Real(60.0) * Real(0.3048) * Real(0.3048));

    // ---- Fuel moistures (Fortran lines 57-62) ----
    Real M[6];
    M[0] = node.m1;
    M[1] = node.m10;
    M[2] = node.m100;
    M[3] = node.m1;
    M[4] = node.mlh;
    M[5] = node.mlw;

    // ---- Live fuel moisture of extinction (Fortran lines 68-75) ----
    Real MPRIMENUMER[4];
    MPRIMENUMER[0] = table.wprimenumer[0] * M[0];
    MPRIMENUMER[1] = table.wprimenumer[1] * M[1];
    MPRIMENUMER[2] = table.wprimenumer[2] * M[2];
    MPRIMENUMER[3] = table.wprimenumer[3] * M[3];
    Real SUM_MPRIMENUMER = MPRIMENUMER[0] + MPRIMENUMER[1] + MPRIMENUMER[2] + MPRIMENUMER[3];
    Real MEX_LIVE = table.mex_live * (Real(1.0) - table.r_mprimedenome14sum_mex_dead * SUM_MPRIMENUMER) - Real(0.226);
    MEX_LIVE = max(MEX_LIVE, table.mex_dead);

    Real FMEX[6];
    FMEX[4] = table.f[4] * MEX_LIVE;
    FMEX[5] = table.f[5] * MEX_LIVE;
    FMEX[0] = table.fmex[0];
    FMEX[1] = table.fmex[1];
    FMEX[2] = table.fmex[2];
    FMEX[3] = table.fmex[3];

    // ---- FMC, QIG, FEPSQIG (Fortran lines 77-82) ----
    Real FMC[6], QIG[6], FEPSQIG[6];
    for (int i = 0; i < 6; i++) {
        FMC[i]     = table.f[i] * M[i];
        QIG[i]     = Real(250.0) + Real(1116.0) * M[i];
        FEPSQIG[i] = table.feps[i] * QIG[i];
    }

    // ---- Heat sink (Fortran lines 83-85) ----
    Real RHOBEPSQIG_DEAD = table.rhob * (FEPSQIG[0] + FEPSQIG[1] + FEPSQIG[2] + FEPSQIG[3]);
    Real RHOBEPSQIG_LIVE = table.rhob * (FEPSQIG[4] + FEPSQIG[5]);
    Real RHOBEPSQIG      = table.f_dead * RHOBEPSQIG_DEAD + table.f_live * RHOBEPSQIG_LIVE;

    // ---- Dead fuel moisture damping (Fortran lines 87-93) ----
    Real M_DEAD = FMC[0] + FMC[1] + FMC[2] + FMC[3];
    Real MOMEX  = M_DEAD / table.mex_dead;
    Real MOMEX2 = MOMEX * MOMEX;
    Real MOMEX3 = MOMEX2 * MOMEX;
    Real ETAM_DEAD = Real(1.0) - Real(2.59)*MOMEX + Real(5.11)*MOMEX2 - Real(3.52)*MOMEX3;
    ETAM_DEAD = max(Real(0.0), min(ETAM_DEAD, Real(1.0)));
    Real IR_DEAD = table.gp_wnd_emd_es_hoc * ETAM_DEAD;

    // ---- Live fuel moisture damping (Fortran lines 95-101) ----
    Real M_LIVE = FMC[4] + FMC[5];
    MOMEX  = M_LIVE / MEX_LIVE;
    MOMEX2 = MOMEX * MOMEX;
    MOMEX3 = MOMEX2 * MOMEX;
    Real ETAM_LIVE = Real(1.0) - Real(2.59)*MOMEX + Real(5.11)*MOMEX2 - Real(3.52)*MOMEX3;
    ETAM_LIVE = max(Real(0.0), min(ETAM_LIVE, Real(1.0)));
    Real IR_LIVE = table.gp_wnl_eml_es_hoc * ETAM_LIVE;

    Real IR = IR_DEAD + IR_LIVE;

    // ---- Wind factor (Fortran lines 105-109) ----
    constexpr Real MPH_TO_FTPMIN = Real(88.0);
    Real WS_LIMIT          = Real(0.9) * IR;
    Real WSMF_LIMITED      = min(node.wsmf * MPH_TO_FTPMIN, WS_LIMIT);
    node.phiw_surface      = table.phiwterm * std::pow(WSMF_LIMITED, table.b_coeff);

    // ---- Slope factor (Fortran lines 111-113) ----
    Real PHIS_MAX     = table.phiwterm * std::pow(WS_LIMIT, table.b_coeff);
    node.phis_surface = min(table.phisterm * node.tanslp2, PHIS_MAX);

    // ---- No-wind/no-slope spread rate (Fortran lines 116-118) ----
    node.vs0 = node.adj * IR * table.xi / RHOBEPSQIG;

    // ---- Direction of max spread velocity (Fortran line 120) ----
    node.velocity_dms_surface = node.vs0 * (Real(1.0) + node.phis_surface + node.phiw_surface);

    // ---- SI unit conversions (Fortran lines 122-125) ----
    node.ir               = IR * BTUPFT2MIN_TO_KWPM2;
    node.hpua_surface     = node.ir * table.tr * Real(60.0);
    node.flin_dms_surface = table.tr * node.ir * node.velocity_dms_surface * Real(0.3048);

    // ---- Debug print (CPU only — stripped on GPU device) ----
#ifndef __CUDA_ARCH__
    static bool first_vs0_calc = true;
    if (first_vs0_calc) {
        amrex::Print() << "\n=== C++ Rothermel vs0 Calculation ===\n";
        amrex::Print() << "IR = " << IR << " Btu/ft²/min\n";
        amrex::Print() << "xi = " << table.xi << "\n";
        amrex::Print() << "RHOBEPSQIG = " << RHOBEPSQIG << "\n";
        amrex::Print() << "vs0 = " << node.vs0 << " ft/min\n";
        amrex::Print() << "====================================\n\n";
        first_vs0_calc = false;
    }
#endif
}

#endif // ELMFIRE_ROTHERMEL_H
