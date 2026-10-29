#ifndef ELMFIRE_FIRENODEDATA_H
#define ELMFIRE_FIRENODEDATA_H

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>

/**
 * FireNodeData: GPU-safe POD struct for a single narrow-band fire cell.
 *
 * Identical fields to FireNode but without linked-list next/prev pointers,
 * allowing storage in flat arrays (amrex::Gpu::DeviceVector) and use inside
 * AMREX_GPU_DEVICE kernels. FireNode remains the CPU-side struct; this is
 * the device-side counterpart introduced for GPU porting.
 */
struct FireNodeData {

    // Grid location
    int ix = 0;
    int iy = 0;

    // Level set field
    amrex::Real phi       = 1000.0;
    amrex::Real phip_old  = 1000.0;
    amrex::Real phip_rk1  = 1000.0;
    amrex::Real ux        = 0.0;
    amrex::Real uy        = 0.0;

    // Gradients & normal vectors
    amrex::Real dphidx          = 0.0;
    amrex::Real dphidy          = 0.0;
    amrex::Real dphidx_limited  = 0.0;
    amrex::Real dphidy_limited  = 0.0;
    amrex::Real gradmag         = 1.0;
    amrex::Real normx           = 0.0;
    amrex::Real normy           = 1.0;
    amrex::Real normx_dms       = 1.0;
    amrex::Real normy_dms       = 0.0;

    // Weather
    amrex::Real ws20   = 0.0;
    amrex::Real wd20   = 0.0;
    amrex::Real u_wind = 0.0;
    amrex::Real v_wind = 0.0;
    amrex::Real wsmf   = 0.0;

    // Fuel moisture
    amrex::Real m1   = 0.06;
    amrex::Real m10  = 0.08;
    amrex::Real m100 = 0.10;
    amrex::Real mlh  = 0.50;
    amrex::Real mlw  = 1.00;

    // Terrain
    amrex::Real slope_x = 0.0;
    amrex::Real slope_y = 0.0;
    amrex::Real tanslp2 = 0.0;

    // Rothermel outputs
    amrex::Real vs0                  = 0.0;
    amrex::Real phiw_surface         = 0.0;
    amrex::Real phis_surface         = 0.0;
    amrex::Real velocity_dms_surface = 0.0;
    amrex::Real vback                = 0.0;
    amrex::Real ir                   = 0.0;
    amrex::Real hpua_surface         = 0.0;
    amrex::Real flin_dms_surface     = 0.0;
    amrex::Real flame_length         = 0.0;

    // Elliptical fire
    amrex::Real phisx = 0.0;
    amrex::Real phisy = 0.0;
    amrex::Real low   = 1.0;

    // Fuel model
    int         ifbfm = 1;
    amrex::Real adj   = 1.0;

    // Fuel model parameters
    amrex::Real b_coeff   = 2.0;
    amrex::Real phiwterm  = 1.0;
    amrex::Real phisterm  = 5.275;

    // Slope correction
    amrex::Real uxousx = 1.0;
    amrex::Real uyousy = 1.0;

    // Tracking
    amrex::Real time_added = 0.0;
    bool        burned     = false;
    bool        front      = false;

    // Fuel fraction / burnout
    amrex::Real time_of_ignition   = -1.0;
    amrex::Real fuel_frac          = 1.0;
    amrex::Real burnout_time       = 600.0;
    amrex::Real burnout_time_table = 600.0;
    amrex::Real flux_scaling       = 1.0;

    // Crown fire
    int crown_fire = 0;

    // Spotting (docs/spotting_S1_design.md)
    amrex::Real flin_at_ignition = 0.0;  // local fireline intensity when the cell burned [kW/m]
    int         spot_emitter     = -1;   // 1 emits embers in its emission window, 0 not, -1 not decided yet

    // Constructors
    AMREX_GPU_HOST_DEVICE FireNodeData() = default;

    AMREX_GPU_HOST_DEVICE FireNodeData(int i, int j, amrex::Real t)
        : ix(i), iy(j), time_added(t) {}

    // Utility methods
    AMREX_GPU_HOST_DEVICE amrex::Real GetVelocityMagnitude() const {
        return std::sqrt(ux * ux + uy * uy);
    }

    AMREX_GPU_HOST_DEVICE amrex::Real GetWindMagnitude() const {
        return std::sqrt(u_wind * u_wind + v_wind * v_wind);
    }

    AMREX_GPU_HOST_DEVICE bool IsActive() const {
        return !burned && amrex::Math::abs(phi) < amrex::Real(1000.0);
    }
};

#endif // ELMFIRE_FIRENODEDATA_H
