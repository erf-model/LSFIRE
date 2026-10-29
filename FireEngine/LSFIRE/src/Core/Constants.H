#ifndef ELMFIRE_CONSTANTS_H
#define ELMFIRE_CONSTANTS_H

#include <AMReX_REAL.H>

namespace ElmfireConstants {

// Mathematical constants (match ELMFIRE_VARS.f90)
constexpr amrex::Real PI = 3.141592653589793;
constexpr amrex::Real PIO180 = PI / 180.0;  // Degrees to radians

// Unit conversions
constexpr amrex::Real FT_TO_M = 0.3048;           // Feet to meters
constexpr amrex::Real M_TO_FT = 1.0 / FT_TO_M;   // Meters to feet
constexpr amrex::Real MPH_TO_MS = 0.44704;        // Miles/hour to m/s
constexpr amrex::Real MS_TO_MPH = 1.0 / MPH_TO_MS;
constexpr amrex::Real FPM_TO_MS = FT_TO_M / 60.0; // Feet/min to m/s
constexpr amrex::Real MS_TO_FPM = 1.0 / FPM_TO_MS;

// Energy conversions
constexpr amrex::Real BTU_TO_KJ = 1.055;          // BTU to kilojoules
constexpr amrex::Real BTUPFT2MIN_TO_KWPM2 = 1.055 / (60.0 * 0.3048 * 0.3048);

// Area conversions
constexpr amrex::Real M2_TO_ACRES = 0.000247105;  // Square meters to acres

// Physical constants
constexpr amrex::Real RHO_PARTICLE = 32.0;        // Fuel particle density (lb/ft³)
constexpr amrex::Real HEAT_OF_COMBUSTION = 8000.0; // BTU/lb (typical)

// Numerical parameters
constexpr amrex::Real SMALL_VALUE = 1.0e-10;
constexpr amrex::Real LARGE_VALUE = 1.0e10;

} // namespace ElmfireConstants

#endif
