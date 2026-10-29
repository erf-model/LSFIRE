#ifndef ELMFIRE_ENGINE_H
#define ELMFIRE_ENGINE_H

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Geometry.H>
#include "FireNode.H"
#include "PhiCorrectionStats.H"
#include "../DataStructures/GpuNarrowBand.H"
#include "../Physics/FuelModels.H"
#include "../Physics/Spotting.H"
#include <fstream>
#include <map>
#include <memory>
#include <unordered_set>
#include <vector>

/**
 * @file ElmfireEngine.H
 * @brief Main ELMFIRE simulation engine
 *
 * This class integrates all ELMFIRE components into a complete fire simulation:
 * - Physics: Rothermel spread rate, elliptical fire
 * - Numerics: Gradient calculation, flux limiting, RK2 integration
 * - Data management: Narrow band tracking, level set evolution
 */

/**
 * @brief Host copy of phi over the bounding box of this rank's FABs (ghosts included),
 *        for neighbour lookups of narrow-band nodes. Cells outside it read as 1000
 *        (unburned). Sized by the local boxes, not the level domain: a static fire L1
 *        covers a small part of a large domain, and a domain-sized buffer refilled every
 *        substep dominated the run time.
 */
struct LocalPhiWindow {
    int ilo = 0, jlo = 0, nx = 0, ny = 0;
    std::vector<amrex::Real> v;   // kept between calls to avoid reallocating every step

    void Fill (const amrex::MultiFab& phi) {
        amrex::Box bb;
        for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
            bb = bb.isEmpty() ? mfi.fabbox() : bb.minBox(mfi.fabbox());
        }
        if (bb.isEmpty()) { nx = ny = 0; return; }
        ilo = bb.smallEnd(0); jlo = bb.smallEnd(1);
        nx = bb.length(0);    ny = bb.length(1);
        v.assign(static_cast<std::size_t>(nx) * ny, 1000.0);
        for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
            const amrex::Box& fb = mfi.fabbox();
            const auto& a = phi.const_array(mfi);
            for (int j = fb.smallEnd(1); j <= fb.bigEnd(1); ++j)
                for (int i = fb.smallEnd(0); i <= fb.bigEnd(0); ++i)
                    v[static_cast<std::size_t>(j - jlo) * nx + (i - ilo)] = a(i, j, 0);
        }
    }

    amrex::Real operator() (int i, int j) const {
        const int ii = i - ilo, jj = j - jlo;
        if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) return 1000.0;
        return v[static_cast<std::size_t>(jj) * nx + ii];
    }
};

class ElmfireEngine {
public:
    /**
     * @brief Constructor
     */
    ElmfireEngine();

    /**
     * @brief Destructor
     */
    ~ElmfireEngine();

    /**
     * @brief Initialize simulation domain and parameters
     *
     * @param geom AMReX geometry (defines domain size and resolution)
     * @param fuel_database Fuel model database
     */
    void Initialize(const amrex::Geometry& geom,
                    FuelModelDatabase& fuel_database,
                    const amrex::BoxArray* ba_in = nullptr,
                    const amrex::DistributionMapping* dm_in = nullptr);

    /**
     * @brief Set initial ignition location
     *
     * @param x X-coordinate of ignition point (m)
     * @param y Y-coordinate of ignition point (m)
     * @param radius Initial fire radius (m)
     * @param time Ignition time (s)
     */
    void SetIgnition(amrex::Real x, amrex::Real y,
                     amrex::Real radius, amrex::Real time);

    /**
     * @brief Set area ignition from mask
     *
     * @param ignition_mask iMultiFab containing ignition mask (0=unignited, 1=ignited)
     */
    void SetAreaIgnition(const amrex::iMultiFab* ignition_mask);

    /**
     * @brief Set fuel model source type
     *
     * @param source "uniform", "metgrid_igbp", or "landfire"
     */
    void SetFuelModelSource(const std::string& source);

    /**
     * @brief Set fuel model field
     *
     * @param fuel_model_id Fuel model ID for entire domain (uniform)
     */
    void SetUniformFuel(int fuel_model_id);

    /**
     * @brief Set uniform weather conditions
     *
     * @param wind_speed Wind speed at 20-ft (mph)
     * @param wind_direction Wind direction (degrees from north, meteorological)
     * @param m1 1-hour fuel moisture (fraction)
     * @param m10 10-hour fuel moisture (fraction)
     * @param m100 100-hour fuel moisture (fraction)
     * @param mlh Live herbaceous moisture (fraction)
     * @param mlw Live woody moisture (fraction)
     */
    void SetUniformWeather(amrex::Real wind_speed,
                          amrex::Real wind_direction,
                          amrex::Real m1,
                          amrex::Real m10,
                          amrex::Real m100,
                          amrex::Real mlh,
                          amrex::Real mlw);

    /**
     * @brief Set uniform slope conditions
     *
     * @param slope_x Terrain slope in x-direction (rise/run, dimensionless)
     * @param slope_y Terrain slope in y-direction (rise/run, dimensionless)
     */
    void SetUniformSlope(amrex::Real slope_x, amrex::Real slope_y);

    /**
     * @brief Advance simulation by one timestep (constant wind mode)
     *
     * Main simulation loop that:
     * 1. Updates Rothermel spread rates for all nodes
     * 2. Computes elliptical fire velocities
     * 3. Calculates gradients and normal vectors
     * 4. Applies flux limiters
     * 5. Performs RK2 time integration (2 substeps)
     * 6. Updates narrow band
     *
     * @param dt Timestep size (s) - will be modified to satisfy CFL
     * @return Actual timestep used (may be smaller than input)
     */
    amrex::Real Advance(amrex::Real dt);

    /**
     * @brief Advance fire by a fixed interval, sub-cycling if CFL requires smaller steps.
     *
     * Calls Advance() in a loop until the total elapsed time reaches fire_dt.
     * Each sub-step uses the CFL-stable dt from the current narrow band; the
     * last sub-step is trimmed so the total is exactly fire_dt.
     *
     * @param fire_dt  Desired total advance interval (s)
     * @return Total time advanced (= fire_dt)
     */
    amrex::Real AdvanceInterval(amrex::Real fire_dt);

    /**
     * @brief Advance simulation by one timestep (atmospheric coupling mode)
     *
     * Advances fire with wind/terrain from ERF atmosphere:
     * 1. Extracts atmospheric wind from MultiFabs
     * 2. Scales wind using logarithmic profile
     * 3. Updates Rothermel spread rates with atmospheric wind
     * 4. Computes elliptical fire velocities
     * 5. Performs RK2 time integration
     * 6. Updates narrow band
     *
     * @param dt Timestep size (s) - will be modified to satisfy CFL
     * @param wind_u Surface U-velocity MultiFab from ERF (m/s)
     * @param wind_v Surface V-velocity MultiFab from ERF (m/s)
     * @param terrain Terrain height MultiFab from ERF (m)
     * @param luindex Land use index iMultiFab from ERF (optional)
     * @return Actual timestep used (may be smaller than input)
     */
    amrex::Real Advance(amrex::Real dt,
                        const amrex::MultiFab* wind_u,
                        const amrex::MultiFab* wind_v,
                        const amrex::MultiFab* terrain,
                        const amrex::iMultiFab* luindex = nullptr);

    /**
     * @brief Extract atmospheric wind/terrain data into fire nodes (call before Advance)
     *
     * @param wind_u Surface U-velocity MultiFab from ERF (m/s)
     * @param wind_v Surface V-velocity MultiFab from ERF (m/s)
     * @param terrain Terrain height MultiFab from ERF (m)
     * @param luindex Land use index MultiFab from ERF (optional)
     */
    void UpdateAtmosphericData(const amrex::MultiFab* wind_u,
                              const amrex::MultiFab* wind_v,
                              const amrex::MultiFab* terrain,
                              const amrex::iMultiFab* luindex = nullptr,
                              const amrex::MultiFab* wind_height = nullptr);

    /**
     * @brief Get current simulation time
     */
    amrex::Real GetTime() const { return m_time; }

    /**
     * @brief Set current simulation time (for restart)
     * @param time Time to set (s)
     */
    void SetTime(amrex::Real time) { m_time = time; }

    void SetBandThickness(int cells) { m_band_thickness = cells; }
    void SetGridAwareBurnout(bool enable) { m_use_grid_aware_burnout = enable; }
    void SetSourceFuelScaling(bool enable, amrex::Real exponent = 1.0) {
        m_use_source_fuel_scaling = enable;
        m_source_fuel_exponent    = exponent;
    }
    void SetTerrainScaleFactor(amrex::Real f) { m_terrain_scale_factor = f; }

    /**
     * @brief Set wind data scale factor (e.g., HRRR bias correction)
     * @param factor Scale factor (default 0.7)
     */
    void SetWindDataScaleFactor(amrex::Real factor) { m_wind_data_scale_factor = factor; }
    void SetAtmosWindHeight(amrex::Real h) { m_atmos_wind_height = h; }

    /**
     * @brief Restore phi from checkpoint and rebuild narrow band
     * @param phi_restart MultiFab containing phi values from checkpoint
     */
    void RestorePhiFromCheckpoint(const amrex::MultiFab& phi_restart);

    /**
     * @brief Non-destructive multilevel phi sync (Phase 4, static multilevel
     * plan). Copies valid + ghost phi from a hierarchy-side MultiFab
     * (assumed to share this engine's BoxArray/DistributionMapping — same
     * layout copy, not a redistribution) directly into m_phi, then
     * refreshes ghost cells.
     *
     * Deliberately does NOT call TagNarrowBand() and does NOT touch any
     * existing narrow-band node's fuel_frac/time_of_ignition/burnout_time.
     * This is the whole point versus RestorePhiFromCheckpoint(): use this
     * for ordinary per-step coarse<->fine synchronization; use
     * RestorePhiFromCheckpoint() only for the rare from-scratch level
     * creation case where there is no prior state to preserve anyway.
     * See docs/LSFIRE_static_multilevel_plan.md KS-HISTORY-001.
     *
     * Cells whose burn classification changes as a result of the synced
     * phi are NOT reconciled by this call: the narrow band is left as-is.
     * A zero-time contract test confirms that changing a valid cell from
     * unburned to burned leaves it without a node even after one Advance().
     * Do not use this phi-only API for routine fine-to-coarse feedback until
     * a separate history-aware node transition is implemented and tested.
     *
     * @param phi_src Hierarchy-side phi MultiFab, same BoxArray/DM as this
     *                engine's own m_phi.
     *
     * NOT called from the per-step AdvanceLevel loop as of 2026-09-24 (see
     * docs/DEVELOPMENT_LOG_2026-09-24.md) — a full 6h dynamic-AMR run
     * showed an unexplained fire_area regression when this was used for
     * routine coarse<->fine sync every step. Use FillGhostPhiFromHierarchy()
     * below for that. This function is kept for a future, explicit
     * valid-region-replacement scenario (not yet designed) where actually
     * overwriting valid cells is the intended operation.
     */
    void SyncPhiFromHierarchy(const amrex::MultiFab& phi_src);

    /**
     * @brief Apply a monotone valid-cell phi correction and reconcile nodes.
     *
     * Copies the candidate into persistent managed scratch, then applies
     * min(old_phi, candidate_phi) on valid cells only. Newly burned cells
     * activate valid narrow-band nodes at sync_time with fresh fuel history;
     * existing burned nodes retain their fuel/ignition/burnout history.
     * This does not choose the fine-to-coarse restriction operator and is
     * called by the opt-in static feedback experiment.
     */
    void ApplyValidPhiCorrection(const amrex::MultiFab& candidate,
                                 amrex::Real sync_time,
                                 int edge_jlo = -1, int edge_jhi = -1);
    const ELMFIRE::PhiCorrectionStats& GetLastPhiCorrectionStats() const
    { return m_last_phi_correction_stats; }

    /**
     * @brief Ghost-only multilevel phi fill (Phase 4, narrowed scope,
     * 2026-09-24 — see docs/DEVELOPMENT_LOG_2026-09-24.md).
     *
     * Updates ONLY the ghost cells of m_phi from a hierarchy-side MultiFab
     * (typically the output of FillPatchTwoLevels, so ghost cells reflect
     * coarse-interpolated data at coarse/fine interfaces and same-level
     * neighbor data elsewhere). The valid region of m_phi is left EXACTLY
     * as it was before the call — implemented by snapshotting the valid
     * region first, copying the full (valid+ghost) source over m_phi, then
     * restoring the snapshot into the valid region, so "valid unchanged"
     * holds by construction rather than by assumption about what
     * FillPatchTwoLevels does with already-valid fine data.
     *
     * Like SyncPhiFromHierarchy(), does NOT call TagNarrowBand() and does
     * NOT touch any narrow-band node's fuel_frac/time_of_ignition/
     * burnout_time/burned — ghost cells are stencil/boundary data, not
     * additional owned burning cells (see the static multilevel plan,
     * Phase 5 notes on ownership).
     *
     * @param phi_src Hierarchy-side phi MultiFab (valid+ghost), same
     *                BoxArray/DM as this engine's own m_phi.
     */
    void FillGhostPhiFromHierarchy(const amrex::MultiFab& phi_src);

    /**
     * @brief Regrid: redistribute MultiFabs and migrate narrow-band nodes
     *
     * Called during AMR regridding when the BoxArray/DistributionMapping change.
     * Instead of destroying and recreating the engine (expensive), this method:
     *   1. Redistributes m_phi and m_fuel_model to the new BA/DM
     *   2. Migrates narrow-band nodes to the rank that owns their cell
     *   3. Preserves all per-node state (fuel_frac, time_of_ignition, etc.)
     *
     * @param new_ba New BoxArray for this level
     * @param new_dm New DistributionMapping for this level
     */
    void Regrid(const amrex::BoxArray& new_ba,
                const amrex::DistributionMapping& new_dm);

    /**
     * @brief Get level set field (for visualization)
     */
    const amrex::MultiFab& GetPhi() const { return m_phi; }
    const amrex::MultiFab& GetPhiField() const { return m_phi; }

    // Persistent physical history on the engine's own grid. Component order
    // is stable for checkpoint/plotfile consumers.
    enum FireHistoryComp { FuelFrac = 0, IgnitionTime, BurnoutTime,
                           BurnoutTimeTable, FluxScaling, Burned, CrownFire,
                           FlinAtIgnition, SpotEmitter,
                           NumFireHistoryComp };
    const amrex::MultiFab& GetFireHistoryField() const { return m_fire_history; }

    /**
     * @brief Get narrow band tracker (for visualization)
     */
    const GpuNarrowBand& GetNarrowBand() const { return m_narrow_band; }

    /**
     * @brief Get narrow band tracker (non-const for UntagCells)
     */
    GpuNarrowBand& GetNarrowBand() { return m_narrow_band; }

    // ------------------------------------------------------------------------
    // Spotting (stage S1, fire level 0; docs/spotting_S1_design.md)
    // ------------------------------------------------------------------------
    const Spotting::Params& GetSpottingParams() const { return m_spot; }
    /** Embers landed per cell (times the sampling factor), since the start. */
    const amrex::MultiFab& GetEmberCount() const { return m_ember_count; }
    /** 1 where a spot fire was ignited. */
    const amrex::MultiFab& GetSpotIgnition() const { return m_spot_ignited; }
    /** Largest ember distance [m] the current front can produce (all ranks; collective). */
    amrex::Real MaxSpottingDistance() const;

    /**
     * @brief Get velocity field from node data (for visualization)
     *
     * @param velocity_u Output: U-component of velocity
     * @param velocity_v Output: V-component of velocity
     */
    void GetVelocityField(amrex::MultiFab& velocity_u,
                         amrex::MultiFab& velocity_v) const;

    /**
     * @brief Get tagged cells field (for visualization)
     * Creates a binary field: 1 = in narrow band, 0 = outside
     *
     * @param tagged Output: Tagged cells field
     */
    void GetTaggedCells(amrex::MultiFab& tagged) const;

    /**
     * @brief Get number of cells in narrow band
     */
    int GetNumActiveCells() const { return m_narrow_band.GetNumNodes(); }

    /**
     * @brief Get recommended timestep from CFL condition
     */
    amrex::Real ComputeTimestep() const;

private:
    // ========================================================================
    // HELPER FUNCTIONS
    // ========================================================================

    /**
     * @brief Extract atmospheric wind from ERF MultiFabs and populate fire nodes
     *
     * Loops over narrow band nodes and:
     * 1. Extracts U, V wind from ERF MultiFabs
     * 2. Extracts terrain height from ERF
     * 3. Applies logarithmic wind profile scaling
     * 4. Applies Wind Adjustment Factor (WAF)
     * 5. Populates node->wsmf, ws20, wd20 for Rothermel
     *
     * @param wind_u Surface U-velocity MultiFab from ERF (m/s at k=1)
     * @param wind_v Surface V-velocity MultiFab from ERF (m/s at k=1)
     * @param terrain Terrain height MultiFab from ERF (m)
     * @param luindex Land use index iMultiFab from ERF (optional, for fuel mapping)
     */
    void ExtractAtmosphericWind(const amrex::MultiFab* wind_u,
                                const amrex::MultiFab* wind_v,
                                const amrex::MultiFab* terrain,
                                const amrex::iMultiFab* luindex,
                                const amrex::MultiFab* wind_height = nullptr);

    // ========================================================================
    // SIMULATION STATE
    // ========================================================================
    amrex::Real m_time;                    // Current simulation time (s)
    amrex::Geometry m_geom;                // Domain geometry
    FuelModelDatabase* m_fuel_db;          // Fuel model database

    // ========================================================================
    // FIELDS
    // ========================================================================
    amrex::MultiFab m_phi;                 // Level set field (signed distance)
    amrex::MultiFab m_phi_correction;      // Persistent managed sync scratch
    amrex::MultiFab m_fire_history;        // Persistent physical node history
    ELMFIRE::PhiCorrectionStats m_last_phi_correction_stats;
    amrex::MultiFab m_fuel_model;          // Fuel model ID at each cell
    int m_uniform_fuel_id;                 // Uniform fuel model ID (for MPI)
    std::string m_fuel_model_source;       // "uniform", "metgrid_igbp", or "landfire"

    // ========================================================================
    // NARROW BAND
    // ========================================================================
    GpuNarrowBand m_narrow_band;           // Active fire front cells
    int m_band_thickness;                  // Thickness of narrow band (cells)

    // ========================================================================
    // WEATHER (uniform for now)
    // ========================================================================
    amrex::Real m_wind_speed;              // 20-ft wind speed (mph)
    amrex::Real m_wind_direction;          // Wind direction (degrees)
    amrex::Real m_m1, m_m10, m_m100;       // Dead fuel moistures
    amrex::Real m_mlh, m_mlw;              // Live fuel moistures

    // ========================================================================
    // TOPOGRAPHY (uniform for now)
    // ========================================================================
    amrex::Real m_slope_x;                 // Terrain slope in x (rise/run)
    amrex::Real m_slope_y;                 // Terrain slope in y (rise/run)
    amrex::Real m_tanslp2;                 // tan²(slope) for Rothermel

    // ========================================================================
    // WIND DATA SCALING
    // ========================================================================
    amrex::Real m_wind_data_scale_factor;  // Scale factor for input wind data (default 0.7)
    amrex::Real m_atmos_wind_height = 46.875;  // m, height of input wind when no per-cell height is given

    // ========================================================================
    // NUMERICAL PARAMETERS
    // ========================================================================
    amrex::Real m_target_cfl;              // Target CFL number (0.5-0.9)
    amrex::Real m_cellsize;                // Grid cell size (m)
    amrex::Real m_rcellsize;               // Reciprocal cell size (1/m)

    // ========================================================================
    // FIRE SUPPRESSION OPTIONS
    // ========================================================================
    bool m_use_grid_aware_burnout;         // Extend burnout_time by dx/R_spread
    bool m_use_source_fuel_scaling;        // Spreadable factor: suppress spread from dead cells
    amrex::Real m_source_fuel_exponent;    // alpha in fuel_frac^alpha (1.0 = linear)
    amrex::Real m_terrain_scale_factor;    // Slope amplification multiplier (1.0 = normal)
    bool m_has_terrain_slope;              // True when per-cell slope set by ExtractAtmosphericWind

    LocalPhiWindow m_phi_window;           // neighbour-phi lookup buffer (reused every step)

    amrex::Real m_ws20_uniform = 0.0;      // constant-wind mode: 20-ft wind speed (mph), for spotting

    // Spotting state
    struct SpotEmber {
        int ix, iy;                        // landing cell (global index)
        amrex::Real t_arrive;              // arrival time [s]
        int positive;                      // DIRECT: ignites on arrival
    };
    Spotting::Params m_spot;
    long m_step = 0;                               // fire steps taken (random-number counter)
    std::vector<SpotEmber> m_spot_pending;         // landings in this rank's boxes, not yet due
    std::map<long, amrex::Real> m_spot_deposited;  // SIMPLE: cells with embers -> local ignition time (-1: none)
    std::vector<int> m_spot_ignited_now;           // (ix, iy) pairs spot-ignited this step on this rank
    amrex::MultiFab m_ember_count;                 // diagnostics
    amrex::MultiFab m_spot_ignited;

    // Extinction (fire.extinction.*): dead burned cells cannot spread (state on the source cell).
    // Applied in the receiver's stencil: a dead neighbour is read as unburned (+1) and phi may not rise next to
    // dead cells. Validated against exact Huygens solutions in LSFIRE_RUN/runs/extinction_masking_20261001.
    bool m_extinction = false;
    amrex::MultiFab m_dead;                        // 1 = dead burned cell, 0 otherwise (ghosts filled)
    LocalPhiWindow m_dead_window;                  // host lookup of m_dead
    bool m_dead_any = false;
    std::vector<amrex::Real> m_ext_test_box;       // fire.extinction.test_box = x0 y0 x1 y1 [m] (testing hook)
    amrex::Real m_ext_test_time = -1.0;            // fire.extinction.test_time [s]
    bool m_ext_test_done = false;
    long m_spot_launched = 0, m_spot_landed = 0, m_spot_ignitions = 0;  // this rank, cumulative
    std::unique_ptr<std::ofstream> m_spot_dump;    // fire.spotting.dump_embers

    // ========================================================================
    // PRIVATE METHODS
    // ========================================================================

    /**
     * @brief Initialize level set field for ignition
     */
    void InitializePhi();

    /**
     * @brief Tag cells in narrow band around fire front
     */
    // Keep physical fields when narrow-band nodes are removed/rebuilt.
    void SaveNodeHistory();
    void RestoreNodeHistory(FireNodeData& node) const;
    void TagNarrowBand();

    /**
     * @brief Tag all cells within band_thickness of location (ix, iy)
     *
     * Matches Fortran TAG_BAND subroutine (elmfire_level_set.f90:1544)
     * @param phi_arr Array accessor for phi MultiFab
     * @param box Valid box for phi_arr (for MPI safety)
     */
    void TagBand(int ix_loc, int iy_loc, std::unordered_set<int>& tagged_set);

    /**
     * @brief Update physics for all nodes (Rothermel + Elliptical)
     */
    void UpdatePhysics();

    /**
     * @brief Perform one RK2 timestep (both stages)
     * @note dt is passed by reference so CflAndFluxLimiter can reduce it;
     *       the CFL-limited value is returned to Advance() via m_time += dt.
     */
    void IntegrateRK2(amrex::Real& dt);

    /**
     * @brief Update narrow band (add new cells, remove burned cells)
     */
    void UpdateNarrowBand();

    /**
     * @brief Add newly burned cells to narrow band (incremental update)
     * Matches Fortran logic: Find cells that crossed phi=0 and tag their neighbors
     */
    void AddNewlyBurnedCells();
    /** Add this rank's band around burned ghost cells (m_phi ghosts must be filled). */
    void TagBandAroundBurnedGhosts(std::unordered_set<int>& tagged_set);

    /** Spotting: embers due by the start of this step land (DIRECT ignition, SIMPLE deposition and
     *  ignition); ignited cells get phi = -1 and enter the band before UpdatePhysics. */
    void SpotApplyArrivals(amrex::Real dt);
    /** Spotting: emitting cells launch embers over [t0, t0 + dt]; landing records are exchanged so
     *  each rank keeps those in its own boxes (collective). */
    void SpotLaunch(amrex::Real t0, amrex::Real dt);
    /** Ignite one cell owned by this rank (phi = -1, add it and its band). False if not ignitable. */
    bool SpotIgniteCell(int ix, int iy, std::unordered_set<int>& tagged_set);
    /** Extinction: apply the dead-cell rules due at m_time (testing hook: cells burned by test_time inside test_box). */
    void ExtinctionUpdate();
    bool IsDead(int ix, int iy) const {
        if (!m_dead_any) return false;
        const amrex::Real v = m_dead_window(ix, iy); return v > 0.5 && v < 1.5;
    }
    /** Spotting: emitter flag (0/1) of a cell that burned with fire-line intensity flin [kW/m]. */
    int SpotDecideEmitter(long key, amrex::Real flin) const;

    /**
     * @brief Update fuel fraction for all burned cells
     * Linear decay: fuel_frac = max(0, 1 - (t - t_ignition) / burnout_time)
     */
    void UpdateFuelFraction();

    /**
     * @brief Remove burned-out cells from narrow band (fuel_frac ~ 0)
     * Prevents narrow band from growing indefinitely
     */
    void DetagBurnedOutCells();

    /**
     * @brief Copy node data to/from MultiFab
     */
    void SyncNodesToMultiFab();
    void SyncMultiFabToNodes();

    /**
     * @brief MPI-safe helper to get phi value at (ix, iy)
     * Searches all local boxes to find which one contains the cell
     * @param ix X-coordinate
     * @param iy Y-coordinate
     * @return Phi value at (ix, iy), or 1.0 if cell not on this rank
     */
    amrex::Real GetPhiAt(int ix, int iy) const;
};

#endif
