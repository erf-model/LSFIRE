#ifndef ELMFIRE_FIRENODE_H
#define ELMFIRE_FIRENODE_H

#include <AMReX_REAL.H>
#include <cmath>

/**
 * @brief Fire node structure - represents one cell in the narrow band
 *
 * This structure closely matches the Fortran TYPE NODE from ELMFIRE_VARS.f90
 * Contains all physics and numerical state for a single fire cell
 */
struct FireNode {

    // ========================================
    // GRID LOCATION (matches Fortran IX, IY)
    // ========================================
    int ix = 0;                    // X grid index (0-based, unlike Fortran)
    int iy = 0;                    // Y grid index

    // ========================================
    // LEVEL SET FIELD (matches Fortran PHIP, UX, UY)
    // ========================================
    amrex::Real phi = 1000.0;      // Level set value (signed distance)
    amrex::Real phip_old = 1000.0; // Previous timestep value
    amrex::Real phip_rk1 = 1000.0; // RK2 intermediate value

    amrex::Real ux = 0.0;          // X-velocity component (m/s)
    amrex::Real uy = 0.0;          // Y-velocity component (m/s)

    // ========================================
    // GRADIENTS & NORMAL VECTORS (matches Fortran DPHIDX, DPHIDY, NORMX, NORMY)
    // ========================================
    amrex::Real dphidx = 0.0;           // Raw gradient in x
    amrex::Real dphidy = 0.0;           // Raw gradient in y
    amrex::Real dphidx_limited = 0.0;   // Flux-limited gradient
    amrex::Real dphidy_limited = 0.0;   // Flux-limited gradient
    amrex::Real gradmag = 1.0;          // Gradient magnitude

    amrex::Real normx = 0.0;       // Normal vector x-component
    amrex::Real normy = 1.0;       // Normal vector y-component
    amrex::Real normx_dms = 1.0;   // DMS (direction of max spread) normal x
    amrex::Real normy_dms = 0.0;   // DMS normal y

    // ========================================
    // WEATHER (matches Fortran WS20, WD20, WSMF)
    // ========================================
    amrex::Real ws20 = 0.0;        // 20-ft wind speed (mph or m/s depending on context)
    amrex::Real wd20 = 0.0;        // 20-ft wind direction (degrees from north)
    amrex::Real u_wind = 0.0;      // Wind u-component (m/s)
    amrex::Real v_wind = 0.0;      // Wind v-component (m/s)
    amrex::Real wsmf = 0.0;        // Midflame wind speed (mph)

    // ========================================
    // FUEL MOISTURE (matches Fortran M1, M10, M100, MLH, MLW)
    // ========================================
    amrex::Real m1 = 0.06;         // 1-hour dead fuel moisture (fraction)
    amrex::Real m10 = 0.08;        // 10-hour dead fuel moisture
    amrex::Real m100 = 0.10;       // 100-hour dead fuel moisture
    amrex::Real mlh = 0.50;        // Live herbaceous moisture
    amrex::Real mlw = 1.00;        // Live woody moisture

    // ========================================
    // TERRAIN (matches Fortran SLOPE_X, SLOPE_Y, TANSLP2)
    // ========================================
    amrex::Real slope_x = 0.0;     // Terrain slope in x-direction
    amrex::Real slope_y = 0.0;     // Terrain slope in y-direction
    amrex::Real tanslp2 = 0.0;     // tan²(slope) for Rothermel

    // ========================================
    // ROTHERMEL OUTPUTS (matches Fortran VS0, PHIW_SURFACE, PHIS_SURFACE, etc.)
    // ========================================
    amrex::Real vs0 = 0.0;                    // No-wind/no-slope spread rate (ft/min)
    amrex::Real phiw_surface = 0.0;           // Wind factor (dimensionless)
    amrex::Real phis_surface = 0.0;           // Slope factor (dimensionless)
    amrex::Real velocity_dms_surface = 0.0;   // Direction of max spread velocity (ft/min)
    amrex::Real vback = 0.0;                  // Backing fire velocity (ft/min)

    amrex::Real ir = 0.0;                     // Reaction intensity (kW/m²)
    amrex::Real hpua_surface = 0.0;           // Heat per unit area (kJ/m²)
    amrex::Real flin_dms_surface = 0.0;       // Fireline intensity (kW/m)
    amrex::Real flame_length = 0.0;           // Flame length (m)

    // ========================================
    // ELLIPTICAL FIRE (matches Fortran PHISX, PHISY, etc.)
    // ========================================
    amrex::Real phisx = 0.0;       // Slope phi components
    amrex::Real phisy = 0.0;
    amrex::Real low = 1.0;         // Length-to-width ratio parameter

    // ========================================
    // FUEL MODEL (matches Fortran IFBFM, ADJ)
    // ========================================
    int ifbfm = 1;                 // Fuel model ID (1-303)
    amrex::Real adj = 1.0;         // Adjustment factor

    // ========================================
    // FUEL MODEL PARAMETERS (matches Fortran B_COEFF, PHIWTERM, etc.)
    // ========================================
    amrex::Real b_coeff = 2.0;     // Wind exponent
    amrex::Real phiwterm = 1.0;    // Wind coefficient
    amrex::Real phisterm = 5.275;  // Slope coefficient

    // ========================================
    // SLOPE CORRECTION (matches Fortran UXOUSX, UYOUSY)
    // ========================================
    amrex::Real uxousx = 1.0;      // X-velocity slope correction
    amrex::Real uyousy = 1.0;      // Y-velocity slope correction

    // ========================================
    // TRACKING (matches Fortran TIME_ADDED, BURNED)
    // ========================================
    amrex::Real time_added = 0.0;  // When cell was added to narrow band
    bool burned = false;           // Has fire passed through?
    bool front = false;            // Is this a front cell?

    // ========================================
    // FUEL FRACTION (burnout tracking)
    // ========================================
    amrex::Real time_of_ignition = -1.0;  // When phi crossed 0 (s), -1 = not yet ignited
    amrex::Real fuel_frac = 1.0;          // Fuel fraction remaining (1.0=full, 0.0=consumed)
    amrex::Real burnout_time = 600.0;     // Effective burnout time (s); includes grid transit time if enabled
    amrex::Real burnout_time_table = 600.0; // Fuel-model table value (s); fixed per fuel type
    amrex::Real flux_scaling = 1.0;       // = burnout_time_table / burnout_time; conserves total HPUA

    // ========================================
    // CROWN FIRE (matches Fortran CROWN_FIRE)
    // ========================================
    int crown_fire = 0;            // 0=surface, 1=passive, 2=active

    // ========================================
    // LINKED LIST POINTERS (matches Fortran NEXT, PREV)
    // ========================================
    FireNode* next = nullptr;      // Next node in list
    FireNode* prev = nullptr;      // Previous node in list

    // ========================================
    // CONSTRUCTORS
    // ========================================
    FireNode() = default;

    FireNode(int i, int j, amrex::Real t)
        : ix(i), iy(j), time_added(t) {}

    // ========================================
    // UTILITY METHODS
    // ========================================

    /**
     * @brief Get velocity magnitude (m/s)
     */
    amrex::Real GetVelocityMagnitude() const {
        return std::sqrt(ux * ux + uy * uy);
    }

    /**
     * @brief Get wind magnitude (m/s)
     */
    amrex::Real GetWindMagnitude() const {
        return std::sqrt(u_wind * u_wind + v_wind * v_wind);
    }

    /**
     * @brief Check if node is active
     */
    bool IsActive() const {
        return !burned && std::abs(phi) < 1000.0;
    }
};

#endif
