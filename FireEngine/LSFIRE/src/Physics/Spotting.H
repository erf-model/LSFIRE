#ifndef ELMFIRE_SPOTTING_H
#define ELMFIRE_SPOTTING_H

#include <AMReX_REAL.H>
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

/**
 * @file Spotting.H
 * @brief Ember spotting, stage S1: port of ELMFIRE 1.1's modular spotting model
 *        (build/source/elmfire_spotting.f90, LAGRANGIAN accumulation) for fire level 0.
 *
 * Components (ELMFIRE names): GENERATION_MODEL RANDOM | PER-AREA | PER-MW,
 * SPOTTING_DISTANCE_MODEL UNIFORM | LOGNORMAL | EMPIRICAL, IGNITION_MODEL DIRECT | SIMPLE.
 * Trajectory: straight line along the source's 20-ft downwind direction (+-4 deg), which replaces
 * ELMFIRE's cell-by-cell wind update (EMPIRICAL distances are < ~2.75 km).
 *
 * Differences from ELMFIRE, chosen so results do not depend on how often the fire is advanced:
 *  - the surface-spotting percent is drawn once per burning cell (at ignition), not every step;
 *  - RANDOM emits its NEMBERS once per cell (at ignition), not every step of the emission window.
 * fire.spotting.per_step_draws = true restores ELMFIRE's per-step draws.
 * Random numbers are a hash of (seed, global cell, fire step, ember, draw): identical for any MPI layout.
 * Design: docs/spotting_S1_design.md.
 */

namespace Spotting {

enum class Generation { Random, PerArea, PerMW };
enum class Distance   { Uniform, Lognormal, Empirical };
enum class Ignition   { Direct, Simple };

struct Params {
    bool enable = false;
    Generation generation = Generation::PerArea;
    Distance   distance   = Distance::Empirical;
    Ignition   ignition   = Ignition::Direct;
    bool per_step_draws   = false;       // ELMFIRE-style draws every step (frequency dependent)

    // Emission (ELMFIRE defaults)
    amrex::Real tau_embergen          = 6.0;    // emission window after ignition [s]
    amrex::Real critical_flin         = 0.0;    // CRITICAL_SPOTTING_FIRELINE_INTENSITY [kW/m]
    amrex::Real surface_spotting_pct  = 100.0;  // SURFACE_FIRE_SPOTTING_PERCENT [%]
    int         nembers_min           = 30;     // RANDOM
    int         nembers_max           = 1;
    amrex::Real ember_gr              = 1.0e-3; // PER-AREA [embers / m^2 / s]
    amrex::Real ember_gr_per_mw_vege  = 33.3;   // PER-MW [embers per MW per s]; cell power = FLIN * dx
    amrex::Real ember_sampling_factor = 1.0;    // each simulated ember stands for this many

    // Distance
    amrex::Real min_distance            = 0.0;   // UNIFORM [m]
    amrex::Real max_distance            = 0.0;
    amrex::Real mean_spotting_dist      = 0.0;   // LOGNORMAL: MSD = mean * FLIN^flin_exp * WS20^ws_exp [m]
    amrex::Real normalized_dist_variance = 0.0;
    amrex::Real spot_flin_exp           = 0.5;
    amrex::Real spot_ws_exp             = 0.9;
    amrex::Real p_eps                   = 0.01;  // EMPIRICAL truncation quantile 1 - p_eps
    bool        use_customized_pdf      = false; // EMPIRICAL with fixed lognormal parameters
    amrex::Real mu_downwind = 0.0, sigma_downwind = 1.0;
    bool        use_crosswind           = false;
    amrex::Real mu_crosswind = 0.0, sigma_crosswind = 1.0;  // customized-pdf crosswind [m]
    amrex::Real tstop                   = 1200.0; // maximum flight time [s] (TSTOP_SPOT)

    // Ignition
    amrex::Real pign               = 1.0;    // PIGN [%]
    amrex::Real source_fuel_ign_mult = 1.0;  // SOURCE_FUEL_IGN_MULT (uniform for all fuels)
    amrex::Real local_ignition_time = 30.0;  // SIMPLE: tau of local ignition [s]
    amrex::Real cell_ignition_delay = 100.0; // SIMPLE: delay from local ignition to phi = -1 [s]

    std::vector<int> ignitable_fuels;        // fuel codes that spot-ignite although IsBurnableFuel() is false (e.g. 91 urban)

    std::uint64_t seed = 12345;
    std::string dump_embers;                 // if set: per-rank CSV of launched embers (<prefix>_rankNNNN.csv)

    /** Read fire.spotting.* (off unless fire.spotting.enable = true). */
    void Read();
    void Print() const;
};

// ----------------------------------------------------------------------------
// Counter-based random numbers
// ----------------------------------------------------------------------------
inline std::uint64_t SplitMix64(std::uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/** Uniform number in (0, 1), a pure function of its arguments. */
inline double Uniform(std::uint64_t seed, std::uint64_t cell, std::uint64_t step,
                      std::uint64_t ember, std::uint64_t stream)
{
    std::uint64_t h = SplitMix64(seed);
    h = SplitMix64(h ^ cell);
    h = SplitMix64(h ^ step);
    h = SplitMix64(h ^ ember);
    h = SplitMix64(h ^ stream);
    return (static_cast<double>(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);  // 2^-53
}

// Streams (the last argument of Uniform)
enum Stream : std::uint64_t { kEmitter = 1, kCount = 2, kCountFrac = 3, kDistance = 4,
                              kDirection = 5, kCrosswind = 6, kDirectIgnition = 7, kSimpleIgnition = 8 };

// ----------------------------------------------------------------------------
// Distributions (as in ELMFIRE)
// ----------------------------------------------------------------------------
/** ELMFIRE ERFINV_LOCAL: Winitzki approximation, a = 0.147. */
amrex::Real ErfInv(amrex::Real x);

/** Lognormal CDF with the log-mean mu and log-sd sigma. */
amrex::Real LognormCdf(amrex::Real x, amrex::Real mu, amrex::Real sigma);

/** ELMFIRE EMPIRICAL_PDF_PARAMETERS, vegetation branch: ws20 [mph], flin [kW/m] ->
 *  downwind lognormal (mu, sigma) and crosswind normal (mu_cw, sigma_cw [m]). */
void EmpiricalPdfParameters(amrex::Real ws20_mph, amrex::Real flin_kwm,
                            amrex::Real& mu, amrex::Real& sigma,
                            amrex::Real& mu_cw, amrex::Real& sigma_cw);

/** Downwind distribution parameters for the configured distance model. */
void DistanceParameters(const Params& p, amrex::Real ws20_mph, amrex::Real flin_kwm,
                        amrex::Real& mu, amrex::Real& sigma, amrex::Real& mu_cw, amrex::Real& sigma_cw);

/** Sample a downwind distance [m] from u in (0,1) (ELMFIRE EMBER_TRAJECTORY).
 *  EMPIRICAL is truncated at the (1 - p_eps) quantile and rounded to whole cells. */
amrex::Real SampleDistance(const Params& p, amrex::Real mu, amrex::Real sigma,
                           amrex::Real cellsize, double u);

/** Largest distance [m] the configured model can produce for (ws20, flin); for L1 tagging. */
amrex::Real MaxDistance(const Params& p, amrex::Real ws20_mph, amrex::Real flin_kwm, amrex::Real cellsize);

/** Expected (real-valued) number of embers emitted by one cell over dt_emit [s]. */
amrex::Real EmberRate(const Params& p, amrex::Real flin_kwm, amrex::Real cellsize, amrex::Real dt_emit);

/** Non-burnable fuel models (no spot ignition): <= 0, 90-99, fill values. */
inline bool IsBurnableFuel(int ifbfm)
{
    return ifbfm > 0 && !(ifbfm >= 90 && ifbfm <= 99) && ifbfm < 300;
}

/** Fuel a landing ember can ignite: burnable, or listed in fire.spotting.ignitable_fuels. */
inline bool IsSpotIgnitable(const Params& p, int ifbfm)
{
    return IsBurnableFuel(ifbfm) ||
           std::find(p.ignitable_fuels.begin(), p.ignitable_fuels.end(), ifbfm) != p.ignitable_fuels.end();
}

} // namespace Spotting

#endif // ELMFIRE_SPOTTING_H
