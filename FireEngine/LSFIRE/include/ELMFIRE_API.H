#ifndef ELMFIRE_API_H
#define ELMFIRE_API_H

#include <AMReX_MultiFab.H>
#include "PhiCorrectionStats.H"
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>
#include <string>
#include <utility>
#include <vector>

// Forward declaration to avoid exposing internal ElmfireEngine class
namespace amrex {
    class MultiFab;
    class Geometry;
}

class ElmfireEngine;

namespace ELMFIRE {

/**
 * @brief Configuration structure for ELMFIRE fire model
 *
 * Replaces config file reading with programmatic configuration.
 * Used by both standalone and coupled (ERF) versions.
 */
struct Config {
    // Numerics
    amrex::Real cfl = 0.9;
    int band_thickness = 5;

    // Fuel
    std::string fuel_model_source = "uniform";  // "uniform", "metgrid_igbp", or "landfire"
    int uniform_fuel_id = 102;  // Default: GR2 (moderate grass)
    std::string fuel_database_path = "fuel_models.csv";

    // Weather modes
    bool use_constant_wind = false;  // true = offline mode, false = coupled mode

    // Constant wind parameters (used if use_constant_wind = true)
    amrex::Real wind_u = 0.0;          // m/s eastward
    amrex::Real wind_v = 0.0;          // m/s northward
    amrex::Real wind_height = 10.0;    // m (reference height)

    // Fuel moisture (constant for now, could be spatially varying later)
    amrex::Real m1 = 0.03;    // 1-hr fuel moisture (fraction)
    amrex::Real m10 = 0.03;   // 10-hr fuel moisture
    amrex::Real m100 = 0.03;  // 100-hr fuel moisture
    amrex::Real mlh = 0.30;   // Live herbaceous moisture
    amrex::Real mlw = 0.60;   // Live woody moisture

    // Wind data scaling (HRRR bias correction)
    amrex::Real wind_data_scale_factor = 0.7;
    amrex::Real atmos_wind_height = 46.875;  // m above ground of the wind passed to UpdateAtmosphericData
                                             // when no per-cell wind_height is given (legacy value)  // Scale factor for input wind data (e.g., HRRR overestimates)

    // Topography (uniform for now)
    amrex::Real slope_x = 0.0;  // Terrain slope in x (rise/run)
    amrex::Real slope_y = 0.0;  // Terrain slope in y

    // Ignition parameters
    amrex::Real ignition_x = 0.0;
    amrex::Real ignition_y = 0.0;
    amrex::Real ignition_radius = 10.0;
    amrex::Real ignition_time = 0.0;

    // Grid-aware burnout (fire suppression physics, default: off)
    // Extends burnout_time by the fire-front transit time across a grid cell:
    //   tau_eff = tau_table + dx / R_spread
    // flux_scaling = tau_table / tau_eff conserves total HPUA (energy per unit area).
    bool use_grid_aware_burnout = false;

    // Source-fuel spreadable factor (fire suppression fix, default: off)
    // Multiplies fire spread velocity by max(fuel_frac_neighbor)^exponent over
    // burned (phi<0) cardinal neighbors, preventing phantom spread from
    // burned-out cells.  Ghost-cell neighbors treated conservatively as active.
    bool use_source_fuel_scaling = false;
    amrex::Real source_fuel_exponent = 1.0;  // alpha in fuel_frac^alpha:
                                              //   alpha=1.0: linear (default)
                                              //   alpha>1.0: front-loaded (influence fades fast)
                                              //   alpha<1.0: tail-loaded (influence persists longer)

    // Terrain (slope) scaling factor
    // Multiplies phis_surface (slope amplification factor) after Rothermel computation,
    // before EllipticalFireVelocity combines it with wind. Aspect direction preserved.
    amrex::Real terrain_scale_factor = 1.0;  // 1.0 = normal slope effect (default)
                                             //   0.0 = flat terrain (no slope influence)
                                             //   <1.0 = reduced slope influence
                                             //   >1.0 = enhanced slope influence
};

/**
 * @brief Main fire engine class for ERF coupling
 *
 * This is a lightweight wrapper around the internal ElmfireEngine class.
 * It provides a clean interface for ERF while hiding ELMFIRE implementation details.
 *
 * Usage:
 *   ELMFIRE::FireEngine fire;
 *   ELMFIRE::Config config;
 *   config.use_constant_wind = true;
 *   config.wind_u = 5.0;
 *   fire.Initialize(geom, config);
 *   fire.SetIgnition(1500.0, 1500.0, 50.0, 0.0);
 *   fire.Advance(dt);  // Mode 1: constant wind
 *   // OR
 *   fire.Advance(dt, U, V, terrain, fuel_map);  // Mode 2: atmospheric coupling
 */
class FireEngine {
public:
    FireEngine();
    ~FireEngine();

    /**
     * @brief Initialize fire model
     *
     * @param geom AMReX geometry (defines domain, grid spacing)
     * @param config Configuration parameters
     * @param ba_in Optional BoxArray from ERF (for matching decomposition)
     * @param dm_in Optional DistributionMapping from ERF (for matching decomposition)
     */
    void Initialize(const amrex::Geometry& geom, const Config& config,
                    const amrex::BoxArray* ba_in = nullptr,
                    const amrex::DistributionMapping* dm_in = nullptr);

    /**
     * @brief Update atmospheric wind/terrain data (for atmospheric coupling mode)
     *
     * Call this before Advance() to provide current wind/terrain from ERF.
     *
     * @param wind_u Surface U-velocity MultiFab from ERF (m/s)
     * @param wind_v Surface V-velocity MultiFab from ERF (m/s)
     * @param terrain Terrain height MultiFab from ERF (m)
     * @param fuel_map Fuel model ID MultiFab (LANDFIRE or other source) (optional)
     * @param wind_height Height above ground (m) of wind_u/wind_v per cell (optional; same
     *        BoxArray/DistributionMapping as wind_u, at least as many ghost cells).
     *        If null, FireConfig::atmos_wind_height is used everywhere.
     */
    void UpdateAtmosphericData(const amrex::MultiFab* wind_u,
                              const amrex::MultiFab* wind_v,
                              const amrex::MultiFab* terrain,
                              const amrex::iMultiFab* fuel_map = nullptr,
                              const amrex::MultiFab* wind_height = nullptr);

    /**
     * @brief Set ignition point (creates circular burned region)
     *
     * @param x X-coordinate of ignition center (m)
     * @param y Y-coordinate of ignition center (m)
     * @param radius Ignition radius (m)
     * @param time Ignition time (s) - currently must be 0.0
     */
    void SetIgnition(amrex::Real x, amrex::Real y,
                    amrex::Real radius, amrex::Real time);

    /**
     * @brief Set area ignition from NetCDF mask
     *
     * @param ignition_mask iMultiFab containing ignition mask (0=unignited, 1=ignited)
     */
    void SetAreaIgnition(const amrex::iMultiFab* ignition_mask);

    /**
     * @brief Advance fire spread with constant wind (Mode 1: Offline)
     *
     * Uses wind parameters from config.
     *
     * @param dt Timestep (s)
     */
    void Advance(amrex::Real dt);

    /**
     * @brief Advance fire by a fixed interval, sub-cycling if CFL requires smaller steps.
     *
     * Calls the internal Advance() in a loop until total elapsed time = fire_dt.
     * If the CFL-stable dt > fire_dt, a single step of exactly fire_dt is taken.
     * If CFL-stable dt < fire_dt, multiple sub-steps are taken automatically.
     *
     * @param fire_dt  Desired total advance interval (s)
     */
    void AdvanceInterval(amrex::Real fire_dt);

    /**
     * @brief Advance fire spread with atmospheric wind (Mode 2: Coupled)
     *
     * Reads wind from ERF MultiFabs at surface level.
     *
     * @param dt Timestep (s)
     * @param wind_u U-velocity MultiFab (m/s) at surface
     * @param wind_v V-velocity MultiFab (m/s) at surface
     * @param terrain Terrain height MultiFab (m) - optional, can be nullptr
     * @param fuel_map Fuel model ID MultiFab - optional, can be nullptr
     */
    void Advance(amrex::Real dt,
                const amrex::MultiFab* wind_u,
                const amrex::MultiFab* wind_v,
                const amrex::MultiFab* terrain,
                const amrex::iMultiFab* fuel_map);

    /**
     * @brief Get level set field (phi)
     *
     * phi < 0: burned
     * phi = 0: fire front
     * phi > 0: unburned
     *
     * @param phi Output MultiFab (must be pre-allocated with correct BoxArray)
     */
    void GetPhiField(amrex::MultiFab& phi) const;

    /**
     * @brief Get sensible heat flux from fire (W/m²)
     *
     * Based on burning rate and heat of combustion.
     * Typical values: O(10^5 - 10^6) W/m² in active fire regions.
     *
     * @param heat_flux Output MultiFab
     */
    void GetHeatFlux(amrex::MultiFab& heat_flux) const;

    /**
     * @brief Get moisture flux from fire (kg/m²/s)
     *
     * Water vapor released during combustion.
     *
     * @param moisture_flux Output MultiFab
     */
    void GetMoistureFlux(amrex::MultiFab& moisture_flux) const;

    /**
     * @brief Get fire spread velocity field
     *
     * @param vel_u X-component of fire velocity (m/s)
     * @param vel_v Y-component of fire velocity (m/s)
     */
    void GetVelocityField(amrex::MultiFab& vel_u, amrex::MultiFab& vel_v) const;

    /**
     * @brief Get burned area (m²)
     *
     * Counts cells where phi < 0.
     *
     * @return Total burned area across all MPI ranks
     */
    amrex::Real GetFireArea() const;

    /**
     * @brief Get number of cells in narrow band
     *
     * @return Total narrow band size across all MPI ranks
     */
    int GetNarrowBandSize() const;

    /**
     * @brief Get cell indices of all narrow-band nodes (for AMR tagging).
     *
     * Returns a vector of (ix, iy) pairs representing the fine-level cell
     * indices of all active narrow-band nodes on this MPI rank.
     *
     * @return Vector of (ix, iy) pairs
     */
    std::vector<std::pair<int,int>> GetNarrowBandCells() const;

    /**
     * @brief Get plot variable names for ERF output
     *
     * Returns list of variables that can be written to plt files.
     *
     * @return Vector of variable names
     */
    static std::vector<std::string> GetPlotVarNames();

    /**
     * @brief Fill plot variable data for ERF output
     *
     * @param name Variable name (from GetPlotVarNames)
     * @param mf Output MultiFab to fill
     */
    void FillPlotVar(const std::string& name, amrex::MultiFab& mf) const;

    /**
     * @brief Check if fire model is initialized
     */
    bool IsInitialized() const { return m_initialized; }

    /**
     * @brief Set current simulation time (for restart)
     *
     * Synchronizes internal fire model time with external simulation time.
     * Must be called during restart to ensure fire timing is correct.
     *
     * @param time Current simulation time (s)
     */
    void SetTime(amrex::Real time);

    /**
     * @brief Restore phi from checkpoint and rebuild narrow band
     *
     * Used during restart to restore fire state from a checkpoint file.
     * After calling this, the fire engine is ready to continue simulation.
     *
     * @param phi_restart MultiFab containing phi values from checkpoint
     */
    void RestorePhiFromCheckpoint(const amrex::MultiFab& phi_restart);

    /**
     * @brief Non-destructive multilevel phi sync (static multilevel plan,
     * Phase 4). Copies valid+ghost phi from a hierarchy-side MultiFab
     * (same BoxArray/DistributionMapping as this engine) into the engine's
     * internal phi field, without touching the narrow band or resetting
     * any node's fuel_frac/time_of_ignition/burnout_time.
     *
     * NOT used for routine per-step coarse/fine sync as of 2026-09-24 (see
     * docs/DEVELOPMENT_LOG_2026-09-24.md — caused an unexplained fire_area
     * regression under dynamic AMR). Use FillGhostPhiFromHierarchy() for
     * that; use RestorePhiFromCheckpoint() only for from-scratch level
     * creation.
     *
     * @param phi_src Hierarchy-side phi MultiFab, same layout as this engine.
     */
    void SyncPhiFromHierarchy(const amrex::MultiFab& phi_src);

    /** Apply a monotone valid-cell phi correction and reconcile fire history. */
    void ApplyValidPhiCorrection(const amrex::MultiFab& candidate,
                                 amrex::Real sync_time,
                                 int edge_jlo = -1, int edge_jhi = -1);
    PhiCorrectionStats GetLastPhiCorrectionStats() const;

    /**
     * @brief Ghost-only multilevel phi fill (static multilevel plan,
     * Phase 4, narrowed scope 2026-09-24). Updates ONLY the ghost cells of
     * the engine's internal phi from a hierarchy-side MultiFab (typically
     * a FillPatchTwoLevels result); the valid region is left exactly as it
     * was before the call, guaranteed by construction (snapshot/restore),
     * not by assumption. Does not touch the narrow band or any node's
     * history fields.
     *
     * @param phi_src Hierarchy-side phi MultiFab (valid+ghost), same
     *                layout as this engine.
     */
    void FillGhostPhiFromHierarchy(const amrex::MultiFab& phi_src);

    /**
     * @brief Regrid: redistribute internal data to new BoxArray/DistributionMapping
     *
     * Called during AMR regridding when the grid structure changes.
     * Migrates narrow-band nodes to the correct MPI rank via Alltoallv,
     * preserving all per-node state. Much faster than destroying and
     * recreating the fire engine.
     *
     * Works in both standalone (offline) and coupled (ERF) modes.
     *
     * @param new_ba New BoxArray for this level
     * @param new_dm New DistributionMapping for this level
     */
    void Regrid(const amrex::BoxArray& new_ba,
                const amrex::DistributionMapping& new_dm);

private:
    ElmfireEngine* m_engine;  // Pointer to hide implementation
    Config m_config;
    amrex::Geometry m_geom;
    bool m_initialized;

    // Disable copy/move (fire engine owns resources)
    FireEngine(const FireEngine&) = delete;
    FireEngine& operator=(const FireEngine&) = delete;
};

} // namespace ELMFIRE

#endif // ELMFIRE_API_H
