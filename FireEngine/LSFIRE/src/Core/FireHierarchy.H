#ifndef FIRE_HIERARCHY_H
#define FIRE_HIERARCHY_H

#include <AMReX_AmrCore.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_BCRec.H>

#include "ELMFIRE_API.H"

#include <memory>
#include <fstream>
#include <string>
#include <vector>

/**
 * @brief AMReX hierarchy of fire levels shared by standalone and ERF adapters.
 *
 * Owns a FireEngine and persistent state on each level. The hierarchy manages
 * static refinement, subcycling, coarse/fine phi exchange, and composite
 * coverage. Couplers derive from this class to supply atmospheric forcing.
 */
class FireHierarchy : public amrex::AmrCore
{
public:
    FireHierarchy();

    /** @brief Coupled construction (e.g. inside ERF): the level-0 geometry and the
     *  AMR settings are given explicitly, so the host code's amr.* and geometry.*
     *  inputs are not read; only fire.* and fire_amr.* are. Standalone-only inputs
     *  (time.*, output.*, amr.regrid_int) are ignored. */
    FireHierarchy(const amrex::Geometry& level0_geom, const amrex::AmrInfo& amr_info);

    ~FireHierarchy() override;

    // ========================================================================
    // Initialization & Main Loop
    // ========================================================================

    /** @brief Initialize all levels and set initial conditions. */
    virtual void InitData();

    /** @brief As InitData(), but level 0 uses the given BoxArray/DistributionMapping
     *  (e.g. aligned with the host model's grid) instead of AMReX's own chopping.
     *  Finer levels are built as usual (static refinement). */
    void InitData(const amrex::BoxArray& ba0, const amrex::DistributionMapping& dm0);

    /** @brief Advance all levels from the current time to t_end: level-0 steps of the
     *  fire's own CFL size, the last one clipped to land on t_end, finer levels
     *  subcycled as in Evolve(). Returns the number of level-0 steps taken. */
    int AdvanceTo(amrex::Real t_end);

    /** @brief Static refinement mask on the level-0 grid (x-fastest, nx*ny values of
     *  0/1, at least one 1), replacing fire_amr.static_mask_file. Must be called
     *  before InitData; requires refinement_mode=static and max_level=1. */
    void SetStaticRefineMask(const std::vector<unsigned char>& mask, int nx, int ny);

    amrex::Real CurrentTime() const { return m_t_new[0]; }

    /** Multi-level fire plotfile (all fire levels + composite mask) at an explicit path. */
    void WritePlotFile(const std::string& filename);

    /** Fire engine of a level (host-side access for couplers: fluxes, plot variables). */
    ELMFIRE::FireEngine& LevelEngine(int lev) { return *m_fire[lev]; }
    const ELMFIRE::FireEngine& LevelEngine(int lev) const { return *m_fire[lev]; }

    /** @brief Apply the ignition source on one level: at t=0 (initial=true, inside
     *  InitData) or when fire.ignition_time is reached (initial=false). The default
     *  is the point / line ignition from fire.*; couplers may override it (e.g. area
     *  ignition from a mask). Public because it launches device kernels. */
    virtual void IgniteLevel(int lev, amrex::Real time, bool initial);

    /** @brief Area ignition on a level from a mask (1 = ignite) on that level's grids;
     *  used by IgniteLevel instead of the point/line source. The mask must outlive
     *  the ignition (initial, or when fire.ignition_time is reached). */
    void SetIgnitionMask(int lev, const amrex::iMultiFab* mask);

    /** @brief Main time loop: advance to stop_time. */
    void Evolve();

    /** Physical burned area using the finest available level in each cell. */
    amrex::Real GetCompositeBurnedArea() const;

    /** Fine-over-coarse selection mask on this level (1 = selected). */
    amrex::iMultiFab GetCompositeValidMask(int lev) const;

    /** Instantaneous per-level fire fluxes on this level's grid.
     *  Destination must use this level's BoxArray/DistributionMapping and
     *  host-accessible storage, matching FireEngine's current flux contract.
     *  ERF grid remapping and temporal coupling belong to the coupler.
     */
    void GetLevelHeatFlux(int lev, amrex::MultiFab& dst) const;
    void GetLevelMoistureFlux(int lev, amrex::MultiFab& dst) const;

    // ========================================================================
    // AmrCore overrides (required pure virtual functions)
    // ========================================================================

    /** @brief Create a new level from scratch (initialization only). */
    void MakeNewLevelFromScratch(int lev, amrex::Real time,
                                  const amrex::BoxArray& ba,
                                  const amrex::DistributionMapping& dm) override;

    /** @brief Create a new level by interpolating from coarser level. */
    void MakeNewLevelFromCoarse(int lev, amrex::Real time,
                                 const amrex::BoxArray& ba,
                                 const amrex::DistributionMapping& dm) override;

    /** @brief Remake an existing level (regridding). */
    void RemakeLevel(int lev, amrex::Real time,
                      const amrex::BoxArray& ba,
                      const amrex::DistributionMapping& dm) override;

    /** @brief Delete all data at a level. */
    void ClearLevel(int lev) override;

    /** @brief Tag cells for refinement. */
    virtual void ErrorEst(int lev, amrex::TagBoxArray& tags,
                   amrex::Real time, int ngrow) override;

protected:
    // ========================================================================
    // Per-level data (accessible to subclasses)
    // ========================================================================

    /** Fire engine on every active AMR level. */
    amrex::Vector<std::unique_ptr<ELMFIRE::FireEngine>> m_fire;

    // ========================================================================
    // Configuration (read from inputs file; subclasses may override after ctor)
    // ========================================================================

    ELMFIRE::Config m_fire_config;

    // Ignition state (subclasses can set m_ignited=true to suppress base-class
    // point ignition and handle ignition themselves)
    amrex::Real m_ignition_x = 0.0;
    amrex::Real m_ignition_y = 0.0;
    amrex::Real m_ignition_radius = 50.0;
    amrex::Real m_ignition_time = 0.0;
    std::string m_ignition_shape = "point";
    amrex::Real m_ignition_line_xlo = 0.0;
    amrex::Real m_ignition_line_xhi = 0.0;
    amrex::Real m_ignition_line_width = 0.0;
    bool m_ignited = false;

    // ========================================================================
    // Time integration (virtual so subclasses can inject data before Advance)
    // ========================================================================

    /** @brief Advance a single level by dt.
     *
     *  Subclasses override this to call UpdateAtmosphericData() / SetAreaIgnition()
     *  before delegating to the base-class implementation which calls
     *  fire->Advance(dt) and GetPhiField().
     */
    virtual void AdvanceLevel(int lev, amrex::Real time, amrex::Real dt);

    /** @brief The finest AMR level (used for narrow-band-driven ErrorEst
     *  tagging and finest-level accumulators).
     *
     *  Phase 1 note: this is NOT used anymore to decide which level(s) run
     *  fire physics — every level with a non-null m_fire[lev] now advances
     *  independently in AdvanceLevel(). This accessor only remains for the
     *  tagging and finest-level accumulation paths; see
     *  docs/LSFIRE_static_multilevel_plan.md. */
    int FireLevel() const { return finest_level; }

    // ========================================================================
    // Hooks for extra plot variables (override in subclasses)
    // ========================================================================

    /** @brief Append extra variable names for the plotfile.
     *  Extra variables are output at the fire level only; zero at coarser levels. */
    virtual void GetExtraPlotVarNames(amrex::Vector<std::string>& /*names*/) const {}

    /** @brief Fill one extra plot variable MultiFab at the fire level. */
    virtual void FillExtraPlotVar(const std::string& /*name*/,
                                   amrex::MultiFab& /*mf*/) const {}

private:
    // ========================================================================
    // Per-level data
    // ========================================================================

    /** Phi (level-set) at each level. */
    amrex::Vector<amrex::MultiFab> m_phi_new;
    amrex::Vector<amrex::MultiFab> m_phi_old;

    // Persistent first phi=0 crossing time on each level; -1 means unburned.
    // This is grid state and survives narrow-band node removal.
    amrex::Vector<amrex::MultiFab> m_arrival_time;

    /**
     * Phase 4 scratch buffer: ghost-filled phi per level, used to feed
     * FillGhostPhiFromHierarchy() in AdvanceLevel(). Allocated once per level
     * alongside m_phi_new/m_phi_old (in MakeNewLevelFromScratch/
     * MakeNewLevelFromCoarse/RemakeLevel), NOT per-timestep.
     *
     * Root-cause note (found via AddressSanitizer, 2026-09-24): an earlier
     * version of this code allocated a fresh local MultiFab for this
     * purpose inside AdvanceLevel every single substep. At fine levels
     * this runs thousands of times over a run, and the resulting
     * allocate/free churn on AMReX's Arena eventually corrupted unrelated
     * heap allocations — ASan caught the actual failure as a
     * heap-buffer-overflow inside an unrelated FillPlotVar() call, not at
     * the allocation site itself, which is what made it hard to find by
     * inspection. Keeping this as a persistent per-level member (same
     * pattern as m_phi_new/m_phi_old) avoids the churn entirely.
     */
    amrex::Vector<amrex::MultiFab> m_phi_ghost_filled;

    // Persistent scratch for opt-in fine-to-coarse minimum restriction.
    amrex::Vector<amrex::MultiFab> m_phi_min_coarsened;
    amrex::Vector<amrex::MultiFab> m_phi_min_candidate;

    /** Time at each level. */
    amrex::Vector<amrex::Real> m_t_new;
    amrex::Vector<amrex::Real> m_t_old;

    // ========================================================================
    // Time integration
    // ========================================================================

    /** @brief Recursive timeStep with subcycling. */
    void timeStep(int lev, amrex::Real time, amrex::Real dt);

    /** Record first burned-sign crossing in persistent per-level grid state. */
    void RecordArrivalTime(int lev, amrex::Real time, amrex::Real dt);

    /** @brief Compute timestep for a level. */
    amrex::Real ComputeDt(int lev);

    /** @brief Average phi from fine level to coarse level. */
    void AverageDownPhi(int crse_lev);
    void MinDownPhi(int crse_lev);

    // ========================================================================
    // FillPatch
    // ========================================================================

    /** @brief Fill phi at a level (including ghost cells from coarser). */
    void FillPhi(int lev, amrex::Real time, amrex::MultiFab& phi);

    // ========================================================================
    // Output
    // ========================================================================

    /** @brief Write per-level fire fields with a composite validity mask. */
    void WritePlotFile(int step);

    // ========================================================================
    // Configuration (read from inputs file)
    // ========================================================================

    // AMR parameters
    int m_regrid_int = 5;              // Regrid every N coarse steps
    int m_phi_nghost = 3;              // Ghost cells for phi
    amrex::Real m_phi_buffer = 50.0;   // Buffer distance for phi tagging (meters)
    amrex::Real m_fuel_frac_threshold = 0.01;  // Threshold for active burning

    // ========================================================================
    // Refinement mode (Phase 2/3 — static multilevel plan)
    //   dynamic (default): existing narrow-band-following ErrorEst, unchanged.
    //   static: fixed rectangular corridor tagged once at init; no further
    //           regridding (fire_amr.regrid_int is ignored in this mode).
    // Deliberately NOT combined in this phase — see
    // docs/LSFIRE_static_multilevel_plan.md Phase 2/14.
    // ========================================================================
    enum class RefinementMode { Dynamic, Static };
    RefinementMode m_refinement_mode = RefinementMode::Dynamic;
    bool m_static_min_feedback = false;
    bool m_static_preserve_band_width = false;
    bool m_phi_bilinear_interp = false;  // Phase 8 controlled comparison
    int m_static_min_feedback_min_burned = 1;
    bool m_static_min_feedback_sign_only = false;
    std::string m_feedback_diagnostic_file;
    int m_feedback_diagnostic_interval = 100;
    std::ofstream m_feedback_diagnostic_stream;
    ELMFIRE::PhiCorrectionStats m_feedback_diagnostic_accum;

    // Static-mode corridor geometry: a horizontal band |y - y0| < width/2,
    // tagged at every coarse level below max_level. First (simplest)
    // implementation per plan Phase 3 — analytic geometry, not real GIS data.
    amrex::Real m_static_corridor_y0    = 0.0;
    amrex::Real m_static_corridor_width = 0.0;
    amrex::Real m_static_corridor_amplitude = 0.0;  // zero retains straight band
    amrex::Real m_static_corridor_wavelength = 0.0;
    std::string m_static_polyline_file;
    std::vector<amrex::Real> m_static_polyline_segments; // x0,y0,x1,y1 per segment
    std::string m_static_mask_file;
    std::vector<unsigned char> m_static_refine_mask; // x-fast L0 cells, 0/1
    int m_static_mask_nx = 0;

    // Time stepping
    amrex::Real m_stop_time = 300.0;
    amrex::Real m_dt_coarse = 0.0;    // Outer AMR step (0 = auto from CFL; set via time.dt)
    amrex::Real m_fire_dt   = 0.0;    // Fire advance interval (0 = use outer step; set via fire.fire_dt)
    amrex::Real m_cfl = 0.9;
    amrex::Vector<int> m_nsubsteps;    // Subcycles per level
    int m_max_step = 100000;

    // Output
    amrex::Real m_plot_interval = 10.0;
    std::string m_output_dir = "output_amr";
    int m_plot_count = 0;

    // Step counter
    amrex::Vector<int> m_istep;

    // Boundary conditions (for FillPatch)
    amrex::Vector<amrex::BCRec> m_bcrec;

    // ========================================================================
    // Time-averaged L0 fire fluxes from L0's own FireEngine
    // Accumulated as Σ(flux × dt) between plot outputs; divided by m_cum_time
    // at output time to get the interval-average. Reset after each WritePlotFile.
    // ========================================================================
    amrex::MultiFab m_cum_heat_flux;   // W/m²·s accumulated at L0
    amrex::MultiFab m_cum_moist_flux;  // kg/m²/s·s accumulated at L0
    amrex::Real     m_cum_time = 0.0;  // total dt accumulated since last output

    // Time-averaged fire scalars at fire level (L1).
    // These accumulators are RESET whenever the L1 BoxArray changes (regrid),
    // so the averaging window may be shorter than the full output interval.
    // Use m_cum_time_fire (not m_cum_time) to divide for the L1 average.
    amrex::MultiFab m_cum_heat_flux_fire;
    amrex::MultiFab m_cum_moist_flux_fire;
    amrex::BoxArray m_fire_ba_accum;        // cached BA to detect regrid
    amrex::Real     m_cum_time_fire = 0.0;  // Σ(dt) since last regrid

    /** @brief Read all parameters from inputs file. */
    void ReadParameters();
    void ConstructCommon();
    void ValidateRefinementSources() const;  // static mode has a corridor, polyline or mask
    void FinishInitData();                   // fire engines, ignition, accumulators, first plot
    bool m_coupled = false;                  // constructed with explicit geometry (host code)
    amrex::Vector<const amrex::iMultiFab*> m_ignition_masks;  // per level, null = point/line
};

#endif // FIRE_HIERARCHY_H
