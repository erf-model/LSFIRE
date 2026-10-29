#ifndef ELMFIRE_ELLIPTICAL_FIRE_H
#define ELMFIRE_ELLIPTICAL_FIRE_H

/**
 * Elliptical fire front model (Richards 1995).
 *
 * Implementation is a template so the same source compiles for both CPU
 * (FireNode&) and GPU (FireNodeData&) without code duplication.
 * Debug prints are guarded with #ifndef __CUDA_ARCH__ so they are stripped
 * when compiling for GPU device.
 *
 * Reference: Richards 1995, Int. J. Wildland Fire 5(2):63-72.
 */

#include "FuelModels.H"
#include <AMReX_Algorithm.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
#ifndef __CUDA_ARCH__
#include <AMReX_Print.H>
#include <string>
#endif

// PI is defined in ElmfireConstants; use a local constexpr here so this
// header stays self-contained.
namespace Rothermel_detail {
    constexpr amrex::Real PI_VAL = amrex::Real(3.14159265358979323846);
}

template <typename NodeT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void EllipticalFireVelocity(NodeT& node, const FuelModelTable& table,
                            amrex::Real acceleration_factor = amrex::Real(1.0))
{
    using amrex::Real;
    using amrex::max;
    using amrex::min;

    constexpr Real KWPM2_TO_BTUPFT2MIN = Real(60.0) * Real(0.3048) * Real(0.3048) / Real(1.055);
    constexpr Real FTPMIN_TO_MPS       = Real(0.3048) / Real(60.0);
    constexpr Real DEG_TO_RAD          = Rothermel_detail::PI_VAL / Real(180.0);
    constexpr Real MAX_LOW             = Real(8.0);

    // Step 1: Slope components (Fortran 1764-1769)
    Real APHIS      = acceleration_factor * node.phis_surface;
    Real aspect_rad = std::atan2(node.slope_x, node.slope_y);
    node.phisx      = APHIS * std::sin(aspect_rad);
    node.phisy      = APHIS * std::cos(aspect_rad);

    // Surface-to-horizontal projection correction. The Fortran reference uses
    // OMCOSSLPRAD = 1 - cos(slope_angle), a per-cell field derived from the actual DEM
    // slope, which is exactly 0 (no correction) when there is no slope. That was
    // previously hardcoded here as a literal 0.05, so it no longer vanished for flat
    // terrain -- the degenerate aspect_rad = atan2(0,0) = 0 case still picked up a
    // spurious ~5% velocity bias in whichever component (ux or uy) happened to align
    // with it. slope_x/slope_y are rise/run, so tanslp2 = tan^2(slope_angle) and
    // cos(slope_angle) = 1/sqrt(1+tanslp2); this reduces to 0 exactly when tanslp2 = 0.
    Real omcosslprad = Real(1.0) - Real(1.0) / std::sqrt(Real(1.0) + node.tanslp2);
    node.uxousx     = Real(1.0) - amrex::Math::abs(std::sin(aspect_rad)) * omcosslprad;
    node.uyousy     = Real(1.0) - amrex::Math::abs(std::cos(aspect_rad)) * omcosslprad;

    // Step 2: Wind components and combined wind+slope (Fortran 1780-1793)
    Real APHIW        = acceleration_factor * node.phiw_surface;
    Real wind_dir_rad = (node.wd20 - Real(180.0)) * DEG_TO_RAD;
    Real PHIWX        = APHIW * std::sin(wind_dir_rad);
    Real PHIWY        = APHIW * std::cos(wind_dir_rad);
    Real PHIX         = node.phisx + PHIWX;
    Real PHIY         = node.phisy + PHIWY;

    // Step 3: Direction of maximum spread (Fortran 1795-1803)
    Real PHIMAG = max(std::sqrt(PHIX*PHIX + PHIY*PHIY), Real(1.0e-10));
    if (PHIMAG < Real(1.1e-10)) {
        node.normx_dms = Real(1.0);
        node.normy_dms = Real(0.0);
    } else {
        Real RPHIMAG   = Real(1.0) / PHIMAG;
        node.normx_dms = RPHIMAG * PHIX;
        node.normy_dms = RPHIMAG * PHIY;
    }

    // Step 4: Velocity in DMS direction (Fortran 1804)
    node.velocity_dms_surface = node.vs0 * (acceleration_factor + PHIMAG);

    // Step 5: Effective midflame wind speed (Fortran 1806-1808)
    Real WS_LIMIT = Real(0.9) * node.ir * KWPM2_TO_BTUPFT2MIN;
    Real WSMFEFF  = min(table.wsmfeff_coeff * std::pow(PHIMAG, table.b_coeff_inverse), WS_LIMIT);

    // Step 6: Length-to-width ratio, Anderson (1983), input must be mph
    constexpr Real FTPMIN_TO_MPH = Real(1.0) / Real(88.0);
    Real WSMFEFF_MPH = WSMFEFF * FTPMIN_TO_MPH;
    node.low = Real(0.936) * std::exp( Real(0.2566) * WSMFEFF_MPH)
             + Real(0.461) * std::exp(-Real(0.1548) * WSMFEFF_MPH)
             - Real(0.397);
    node.low = min(max(node.low, Real(1.0)), MAX_LOW);

    // Step 7: Backing fire velocity (Fortran 1812-1818)
    Real BOH;
    if (node.low > Real(0.999) && node.low < Real(1.001)) {
        BOH = Real(1.0);
    } else {
        Real SQRT_LOW2_M1 = std::sqrt(node.low * node.low - Real(1.0));
        BOH = (node.low - SQRT_LOW2_M1) / (node.low + SQRT_LOW2_M1);
    }
    node.vback = BOH * node.velocity_dms_surface;

    // Step 8: Richards ellipse velocity (Fortran 1829-1841)
    Real COSANG   = node.normy * node.normy_dms + node.normx * node.normx_dms;
    Real SINANG   = node.normx * node.normy_dms - node.normy * node.normx_dms;
    Real A        = max(Real(0.5) * (node.velocity_dms_surface + node.vback), Real(1.0e-10));
    Real AACOSANG = A * A * COSANG;
    Real B        = Real(0.5) * max((node.velocity_dms_surface + node.vback) / node.low, Real(1.0e-10));
    Real BBSINANG = B * B * SINANG;
    Real DENOM    = max(std::sqrt(AACOSANG*COSANG + BBSINANG*SINANG), Real(1.0e-10));
    Real RDENOM   = Real(1.0) / DENOM;
    Real DYDT     = RDENOM * AACOSANG + Real(0.5) * (node.velocity_dms_surface - node.vback);
    Real DXDT     = RDENOM * BBSINANG;

    // Step 9: Rotate to X-Y coordinates (Fortran 1844-1848)
    Real DXDT_ROTATED = DYDT * node.normx_dms + DXDT * node.normy_dms;
    node.ux           = DXDT_ROTATED * node.uxousx * FTPMIN_TO_MPS;
    Real DYDT_ROTATED = DYDT * node.normy_dms - DXDT * node.normx_dms;
    node.uy           = DYDT_ROTATED * node.uyousy * FTPMIN_TO_MPS;

    // Step 10: Fireline intensity with actual spread rate (Fortran 1850-1853)
    Real VELOCITY         = std::sqrt(DXDT_ROTATED*DXDT_ROTATED + DYDT_ROTATED*DYDT_ROTATED);
    node.flin_dms_surface = table.tr * node.ir * VELOCITY * Real(0.3048);

    // ---- Debug prints (CPU only) ----
#ifndef __CUDA_ARCH__
    static bool first_wind_debug = true;
    if (first_wind_debug && amrex::Math::abs(node.phi) < Real(10.0)) {
        amrex::Print() << "\nDEBUG EllipticalFireVelocity at (" << node.ix << "," << node.iy << "):\n";
        amrex::Print() << "  APHIW=" << APHIW << " PHIMAG=" << PHIMAG << " vs0=" << node.vs0 << " ft/min\n";
        first_wind_debug = false;
    }

    static int vel_debug_count = 0;
    if (vel_debug_count < 5 && amrex::Math::abs(node.phi) < Real(1.0)) {
        amrex::Print() << "  velocity: (" << node.ux << ", " << node.uy << ") m/s"
                       << "  LOW=" << node.low << "  vback=" << node.vback << " ft/min\n";
        vel_debug_count++;
    }
#endif
}

#endif // ELMFIRE_ELLIPTICAL_FIRE_H
