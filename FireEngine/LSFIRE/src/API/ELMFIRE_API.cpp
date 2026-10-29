#include "ELMFIRE_API.H"
#include "../Core/ElmfireEngine.H"
#include "../Physics/FuelModels.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_iMultiFab.H>
#include <cmath>

namespace ELMFIRE {

FireEngine::FireEngine()
    : m_engine(nullptr)
    , m_initialized(false)
{
}

FireEngine::~FireEngine()
{
    if (m_engine) {
        delete m_engine;
        m_engine = nullptr;
    }
}

void FireEngine::Initialize(const amrex::Geometry& geom, const Config& config,
                            const amrex::BoxArray* ba_in,
                            const amrex::DistributionMapping* dm_in)
{
    using namespace amrex;

    if (m_initialized) {
        amrex::Abort("FireEngine::Initialize() called twice!");
    }

    m_config = config;
    m_geom = geom;

    // Load fuel database
    FuelModelDatabase* fuel_db = new FuelModelDatabase();
    if (!config.fuel_database_path.empty()) {
        fuel_db->LoadFromCSV(config.fuel_database_path);
    }

    // Create internal ELMFIRE engine with optional BoxArray/DistributionMapping from ERF
    m_engine = new ElmfireEngine();
    if (config.band_thickness < 1) {
        amrex::Abort("fire.band_thickness must be positive");
    }
    m_engine->SetBandThickness(config.band_thickness);
    m_engine->Initialize(geom, *fuel_db, ba_in, dm_in);

    // Set fuel model source and uniform fuel
    m_engine->SetFuelModelSource(config.fuel_model_source);
    m_engine->SetUniformFuel(config.uniform_fuel_id);

    // Set uniform slope
    m_engine->SetUniformSlope(config.slope_x, config.slope_y);

    // Set wind data scale factor
    m_engine->SetWindDataScaleFactor(config.wind_data_scale_factor);
    m_engine->SetAtmosWindHeight(config.atmos_wind_height);

    // Fire suppression options
    m_engine->SetGridAwareBurnout(config.use_grid_aware_burnout);
    m_engine->SetSourceFuelScaling(config.use_source_fuel_scaling,
                                   config.source_fuel_exponent);
    m_engine->SetTerrainScaleFactor(config.terrain_scale_factor);

    // Set fuel moisture unconditionally; atmospheric mode overwrites wind each step
    m_engine->SetUniformWeather(0.0, 0.0,
                                config.m1, config.m10, config.m100,
                                config.mlh, config.mlw);

    // Set wind parameters (constant wind mode only)
    if (config.use_constant_wind) {
        // Convert wind components (m/s) to speed and direction for Rothermel
        Real wind_speed_ms = std::sqrt(config.wind_u * config.wind_u +
                                       config.wind_v * config.wind_v);

        // Convert to meteorological "from" direction
        Real wind_direction_math = std::atan2(config.wind_v, config.wind_u) * 180.0 / M_PI;
        Real wind_direction = 270.0 - wind_direction_math;
        if (wind_direction < 0.0) wind_direction += 360.0;
        if (wind_direction >= 360.0) wind_direction -= 360.0;

        // Scale from reference height to 20-ft using log-law
        Real z_ref = config.wind_height;  // Reference height (m)
        Real z_20ft = 6.096;  // 20 ft in meters
        Real z0 = 0.03;  // Roughness length for short grass (m)
        Real wind_speed_20ft_ms = wind_speed_ms * std::log(z_20ft / z0) / std::log(z_ref / z0);
        Real wind_speed_20ft_mph = wind_speed_20ft_ms * 2.23694;  // Convert to mph

        m_engine->SetUniformWeather(wind_speed_20ft_mph, wind_direction,
                                    config.m1, config.m10, config.m100,
                                    config.mlh, config.mlw);
    }

    m_initialized = true;

    if (ParallelDescriptor::IOProcessor()) {
        Print() << "🔥 ELMFIRE FireEngine initialized:\n";
        Print() << "  Domain: " << geom.Domain() << "\n";
        Print() << "  Cell size: " << geom.CellSize()[0] << " m\n";
        Print() << "  Fuel model: " << config.uniform_fuel_id << "\n";
        Print() << "  Wind mode: "
                << (config.use_constant_wind ? "constant" : "atmospheric") << "\n";
        if (config.use_constant_wind) {
            Real wind_speed_ms = std::sqrt(config.wind_u * config.wind_u +
                                          config.wind_v * config.wind_v);
            Print() << "  Wind: (" << config.wind_u << ", " << config.wind_v << ") m/s"
                    << " (speed = " << wind_speed_ms << " m/s)\n";
        }
    }
}

void FireEngine::SetIgnition(amrex::Real x, amrex::Real y,
                             amrex::Real radius, amrex::Real time)
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::SetIgnition() called before Initialize()");
    }

    m_engine->SetIgnition(x, y, radius, time);

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "🔥 Ignition set at (" << x << ", " << y << ") m, "
                       << "radius = " << radius << " m\n";
    }
}

void FireEngine::SetAreaIgnition(const amrex::iMultiFab* ignition_mask)
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::SetAreaIgnition() called before Initialize()");
    }

    m_engine->SetAreaIgnition(ignition_mask);
}

void FireEngine::UpdateAtmosphericData(const amrex::MultiFab* wind_u,
                                      const amrex::MultiFab* wind_v,
                                      const amrex::MultiFab* terrain,
                                      const amrex::iMultiFab* luindex,
                                      const amrex::MultiFab* wind_height)
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::UpdateAtmosphericData() called before Initialize()");
    }

    m_engine->UpdateAtmosphericData(wind_u, wind_v, terrain, luindex, wind_height);
}

void FireEngine::Advance(amrex::Real dt)
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::Advance() called before Initialize()");
    }

    // NOTE: This function now works in both constant wind and atmospheric coupling modes.
    // For atmospheric coupling, call UpdateAtmosphericData() first to set wind data.
    m_engine->Advance(dt);
}

void FireEngine::AdvanceInterval(amrex::Real fire_dt)
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::AdvanceInterval() called before Initialize()");
    }
    m_engine->AdvanceInterval(fire_dt);
}

// Convert ERF land use index to ELMFIRE fuel model ID
static int LUIndexToFuelModel(int lu_index)
{
    switch (lu_index) {
        case 1:  return 165; // Evergreen Needleleaf Forest → TU5
        case 2:  return 145; // Evergreen Broadleaf Forest → TL5
        case 3:  return 161; // Deciduous Needleleaf Forest → TU1
        case 4:  return 161; // Deciduous Broadleaf Forest → TU1
        case 5:  return 162; // Mixed Forest → TU2
        case 6:  return 124; // Closed Shrublands → GS4
        case 7:  return 122; // Open Shrublands → GS2
        case 8:  return 123; // Woody Savannas → GS3
        case 9:  return 123; // Savannas → GS3
        case 10: return 102; // Grasslands → GR2
        case 11: return 0;   // Permanent Wetlands → Non-burnable
        case 12: return 101; // Croplands → GR1
        case 13: return 162; // Urban and Built-Up → TU2
        case 14: return 121; // Cropland/Natural Mosaic → GS1
        case 15: return 0;   // Snow and Ice → Non-burnable
        case 16: return 0;   // Barren or Sparsely Vegetated → Non-burnable
        case 17: return 0;   // Water Bodies → Non-burnable
        case 18: return 102; // Wooded Tundra → GR2
        case 19: return 101; // Mixed Tundra → GR1
        case 20: return 0;   // Barren Tundra → Non-burnable
        case 21: return 0;   // Lake → Non-burnable
        default: return 102; // Default to GR2 (moderate grass)
    }
}

void FireEngine::Advance(amrex::Real dt,
                        const amrex::MultiFab* wind_u,
                        const amrex::MultiFab* wind_v,
                        const amrex::MultiFab* terrain,
                        const amrex::iMultiFab* fuel_map)
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::Advance() called before Initialize()");
    }

    if (m_config.use_constant_wind) {
        amrex::Abort("FireEngine::Advance(dt, U, V, ...) requires use_constant_wind = false");
    }

    // Phase 4: Atmospheric coupling implementation
    // Extract surface wind, terrain, and fuel data from ERF and update narrow band nodes

    GpuNarrowBand& narrow_band = m_engine->GetNarrowBand();
    int num_nodes = narrow_band.GetNumNodes();

    // Get geometry info
    const amrex::Real* dx = m_geom.CellSize();
    const amrex::Real* prob_lo = m_geom.ProbLo();

    // Wind statistics for diagnostic output
    struct {
        amrex::Real u_min = 1e20, u_max = -1e20, u_sum = 0.0;
        amrex::Real v_min = 1e20, v_max = -1e20, v_sum = 0.0;
        amrex::Real wspd_min = 1e20, wspd_max = -1e20, wspd_sum = 0.0;
        amrex::Real z_extract_first = 0.0;  // Record z from first node
        int count = 0;
    } wind_stats;

    for (int n = 0; n < num_nodes; n++) {
        FireNodeData& nd = narrow_band[n];

        int ix = nd.ix;
        int iy = nd.iy;

        // 1. Extract surface wind (U, V) and height if provided
        bool wind_found = false;
        if (wind_u != nullptr && wind_v != nullptr) {
            // Find which box contains this cell
            for (amrex::MFIter mfi(*wind_u); mfi.isValid(); ++mfi) {
                const amrex::Box& box = mfi.validbox();
                if (box.contains(amrex::IntVect(amrex::Dim3{ix, iy, 0}))) {
                    auto u_arr = wind_u->const_array(mfi);
                    auto v_arr = wind_v->const_array(mfi);

                    // ERF extracts k=1 data into the 2D surface array at k=0
                    // So we access k=0 of the 2D array to get k=1 wind from ERF
                    nd.u_wind = u_arr(ix, iy, 0);
                    nd.v_wind = v_arr(ix, iy, 0);
                    wind_found = true;

                    static int wind_debug_count = 0;
                    if (wind_debug_count < 5 && amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Print() << "DEBUG WindExtract node (" << ix << "," << iy << ") found=" << wind_found << ":\n";
                        amrex::Print() << "  u_wind = " << nd.u_wind << " m/s\n";
                        amrex::Print() << "  v_wind = " << nd.v_wind << " m/s\n";
                        amrex::Print() << "  wspd = " << std::sqrt(nd.u_wind*nd.u_wind + nd.v_wind*nd.v_wind) << " m/s\n";
                        wind_debug_count++;
                    }

                    amrex::Real wspd_at_z = std::sqrt(nd.u_wind * nd.u_wind +
                                                        nd.v_wind * nd.v_wind);
                    amrex::Real wdir = std::atan2(-nd.u_wind, -nd.v_wind) * 180.0 / M_PI;
                    if (wdir < 0.0) wdir += 360.0;

                    // Extract actual height from terrain field (if available)
                    amrex::Real z_extract = 46.875;  // Default fallback
                    bool z_found = false;
                    if (terrain != nullptr) {
                        for (amrex::MFIter mfi_terrain(*terrain); mfi_terrain.isValid(); ++mfi_terrain) {
                            const amrex::Box& terrain_box = mfi_terrain.validbox();
                            if (terrain_box.contains(amrex::IntVect(amrex::Dim3{ix, iy, 1}))) {
                                auto z_arr = terrain->const_array(mfi_terrain);
                                z_extract = z_arr(ix, iy, 1);  // Height at k=1 cell center
                                z_found = true;
                                break;
                            }
                        }
                    }

                    // DEBUG: Print z_extract value for first few nodes
                    static int z_debug_count = 0;
                    if (z_debug_count < 5 && amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Print() << "DEBUG Z-extraction node (" << ix << "," << iy << "):\n";
                        amrex::Print() << "  z_extract = " << z_extract << " m (found=" << z_found << ")\n";
                        amrex::Print() << "  wspd_at_z = " << wspd_at_z << " m/s\n";
                        z_debug_count++;
                    }

                    // Apply logarithmic wind profile to scale from extraction height to 20-ft
                    // Log law: U(z) = (u*/κ) * ln(z/z0)
                    // Therefore: U(z1)/U(z2) = ln(z1/z0) / ln(z2/z0)
                    constexpr amrex::Real z_20ft = 6.1;        // 20-ft in meters
                    amrex::Real z0 = 0.03;           // Roughness length for grass (m)

                    // Apply Wind Adjustment Factor (WAF) to convert 20-ft wind to midflame
                    // For grass/shrub fuels without canopy, WAF ≈ 0.4
                    // constexpr amrex::Real WAF = 0.4;
                    // amrex::Real wspd_midflame = wspd_20ft * WAF;

                    // Ye Liu: modify wind adjustment according to fuel model
                    // Apply fuel-specific wind reduction factor
                    amrex::Real wind_adjustment = 0.3;  // Default for typical shrub/grass mix

                    if (nd.ifbfm >= 101 && nd.ifbfm <= 109) {
                        z0 = 0.04;
                        wind_adjustment = 0.42;
                    } else if (nd.ifbfm >= 121 && nd.ifbfm <= 124) {
                        z0 = 0.1;
                        wind_adjustment = 0.38;
                    } else if (nd.ifbfm >= 141 && nd.ifbfm <= 149) {
                        z0 = 0.5;
                        wind_adjustment = 0.27;
                    } else if (nd.ifbfm >= 161 && nd.ifbfm <= 165) {
                        z0 = 1.0;
                        wind_adjustment = 0.32;
                    } else if (nd.ifbfm >= 181 && nd.ifbfm <= 189) {
                        z0 = 1.5;
                        wind_adjustment = 0.11;
                    } else if (nd.ifbfm == 91) {
                        z0 = 1.0;
                        wind_adjustment = 0.30;
                    }
                    // --

                    amrex::Real log_ratio = std::log(z_20ft / z0) / std::log(z_extract / z0);
                    amrex::Real wspd_20ft = wspd_at_z * log_ratio;

                    // High-wind correction: reduce sheltering effect at high wind speeds
                    // Physical basis: vegetation bends/flattens at high winds, reducing canopy sheltering
                    // References:
                    //   - Mueller et al. (2018): "Examination of underlying physics in wildland fire behavior model"
                    //     Field measurements during Santa Ana winds showed wind adjustment factors increase above ~15 m/s
                    //   - Butler et al. (2007): Wind tunnel studies showed sheltering decreases at high winds
                    //   - Andrews (2012): USFS Gen. Tech. Rep. RMRS-GTR-266 (standard approach, constant factors)
                    amrex::Real effective_adjustment = wind_adjustment;  // Start with fuel-specific base value

                    constexpr amrex::Real HIGH_WIND_THRESHOLD = 10.0;  // m/s (~22 mph) - wind speed where sheltering starts to break down
                    if (wspd_20ft > HIGH_WIND_THRESHOLD) {
                        // Gradually increase adjustment factor (reduce sheltering) for winds > 10 m/s
                        // At 20 m/s, sheltering is reduced by ~40%
                        amrex::Real high_wind_multiplier = 1.0 + 0.4 * (wspd_20ft - HIGH_WIND_THRESHOLD) / 10.0;
                        high_wind_multiplier = std::min(high_wind_multiplier, 1.5);  // Cap at 1.5x base adjustment
                        effective_adjustment = std::min(wind_adjustment * high_wind_multiplier, 0.85);  // Never exceed 85% of 20ft wind
                    }

                    // Apply wind data scale factor (e.g., HRRR tends to overestimate wind speed)
                    amrex::Real wspd_midflame = wspd_20ft * effective_adjustment * m_config.wind_data_scale_factor;

                    // Additional safety cap for extreme wind conditions
                    constexpr amrex::Real MAX_MIDFLAME_WIND = 15.0;  // m/s (~34 mph)
                    wspd_midflame = std::min(wspd_midflame, MAX_MIDFLAME_WIND);

                    // Convert m/s to mph for ELMFIRE (Rothermel expects mph)
                    constexpr amrex::Real MS_TO_MPH = 2.23694;

                    nd.ws20 = wspd_20ft * MS_TO_MPH;
                    nd.wsmf = wspd_midflame * MS_TO_MPH;
                    nd.wd20 = wdir;

                    static int scale_debug_count = 0;
                    if (scale_debug_count < 5 && amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Print() << "DEBUG WindScaling node (" << ix << "," << iy << "):\n";
                        amrex::Print() << "  u_wind = " << nd.u_wind << " m/s, v_wind = " << nd.v_wind << " m/s\n";
                        amrex::Print() << "  z_extract = " << z_extract << " m (found=" << z_found << ")\n";
                        amrex::Print() << "  wspd_at_z = " << wspd_at_z << " m/s\n";
                        amrex::Print() << "  log_ratio = ln(" << z_20ft << "/" << z0 << ") / ln(" << z_extract << "/" << z0 << ") = " << log_ratio << "\n";
                        amrex::Print() << "  wspd_20ft = " << wspd_at_z << " × " << log_ratio << " = " << wspd_20ft << " m/s = " << nd.ws20 << " mph\n";
                        amrex::Print() << "  wind_adjustment (base) = " << wind_adjustment << " (fuel-specific)\n";
                        if (effective_adjustment != wind_adjustment) {
                            amrex::Print() << "  effective_adjustment = " << effective_adjustment << " (HIGH-WIND CORRECTION ACTIVE)\n";
                        }
                        amrex::Print() << "  wspd_midflame = " << wspd_20ft << " × " << effective_adjustment << " = " << wspd_midflame << " m/s = " << nd.wsmf << " mph\n";
                        amrex::Print() << "  wdir = " << wdir << " deg (atan2(-u,-v))\n";
                        scale_debug_count++;
                    }

                    wind_stats.u_min = std::min(wind_stats.u_min, nd.u_wind);
                    wind_stats.u_max = std::max(wind_stats.u_max, nd.u_wind);
                    wind_stats.v_min = std::min(wind_stats.v_min, nd.v_wind);
                    wind_stats.v_max = std::max(wind_stats.v_max, nd.v_wind);
                    wind_stats.wspd_min = std::min(wind_stats.wspd_min, wspd_at_z);
                    wind_stats.wspd_max = std::max(wind_stats.wspd_max, wspd_at_z);
                    wind_stats.u_sum += nd.u_wind;
                    wind_stats.v_sum += nd.v_wind;
                    wind_stats.wspd_sum += wspd_at_z;
                    if (wind_stats.count == 0) {
                        wind_stats.z_extract_first = z_extract;  // Record height from first node
                    }
                    wind_stats.count++;

                    break;
                }
            }
        }

        // 2. Extract terrain slope if provided
        if (terrain != nullptr) {
            // Compute terrain slope (dz/dx, dz/dy)
            for (amrex::MFIter mfi(*terrain); mfi.isValid(); ++mfi) {
                const amrex::Box& box = mfi.validbox();
                if (box.contains(amrex::IntVect(amrex::Dim3{ix, iy, 0}))) {
                    auto z_arr = terrain->const_array(mfi);

                    // Central difference for slope
                    amrex::Real z_center = z_arr(ix, iy, 0);
                    amrex::Real z_east = (box.contains(amrex::IntVect(amrex::Dim3{ix+1, iy, 0}))) ?
                                         z_arr(ix+1, iy, 0) : z_center;
                    amrex::Real z_west = (box.contains(amrex::IntVect(amrex::Dim3{ix-1, iy, 0}))) ?
                                         z_arr(ix-1, iy, 0) : z_center;
                    amrex::Real z_north = (box.contains(amrex::IntVect(amrex::Dim3{ix, iy+1, 0}))) ?
                                          z_arr(ix, iy+1, 0) : z_center;
                    amrex::Real z_south = (box.contains(amrex::IntVect(amrex::Dim3{ix, iy-1, 0}))) ?
                                          z_arr(ix, iy-1, 0) : z_center;

                    nd.slope_x = (z_east - z_west) / (2.0 * dx[0]);
                    nd.slope_y = (z_north - z_south) / (2.0 * dx[1]);

                    amrex::Real slope_mag = std::sqrt(nd.slope_x * nd.slope_x +
                                                      nd.slope_y * nd.slope_y);
                    nd.tanslp2 = slope_mag * slope_mag;

                    break;
                }
            }
        }

        // 3. Extract fuel model from lu_index if provided
        if (fuel_map != nullptr) {
            for (amrex::MFIter mfi(*fuel_map); mfi.isValid(); ++mfi) {
                const amrex::Box& box = mfi.validbox();
                if (box.contains(amrex::IntVect(amrex::Dim3{ix, iy, 0}))) {
                    auto fuel_arr = fuel_map->const_array(mfi);
                    int lu_index = fuel_arr(ix, iy, 0);
                    nd.ifbfm = LUIndexToFuelModel(lu_index);
                    break;
                }
            }
        }

    }

    // Now advance fire with updated atmospheric data using atmospheric coupling
    // Call the overloaded Advance() that accepts MultiFab pointers
    m_engine->Advance(dt, wind_u, wind_v, terrain, nullptr);  // Pass nullptr for luindex (handled above)
}

void FireEngine::GetPhiField(amrex::MultiFab& phi) const
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::GetPhiField() called before Initialize()");
    }

    // Copy from internal engine's m_phi MultiFab
    const amrex::MultiFab& engine_phi = m_engine->GetPhi();

    // Use ParallelCopy to handle different BoxArray/DistributionMap between
    // ELMFIRE's internal m_phi and ERF's plot MultiFab (critical for MPI)
    phi.ParallelCopy(engine_phi, 0, 0, 1);
}

void FireEngine::GetHeatFlux(amrex::MultiFab& heat_flux) const
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::GetHeatFlux() called before Initialize()");
    }

    // Initialize to zero
    heat_flux.setVal(0.0);

    // Fraction of total heat released as sensible heat
    // (rest is latent heat from moisture + radiation)
    constexpr amrex::Real f_sensible = 0.75;  // 75% sensible heat (typical for wildfires)

    // Iterate over FABs to handle multi-box decomposition
    for (amrex::MFIter mfi(heat_flux); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        auto heat_arr = heat_flux.array(mfi);

        const GpuNarrowBand& nb = m_engine->GetNarrowBand();
        int num_nodes = nb.GetNumNodes();

        for (int i = 0; i < num_nodes; i++) {
            const FireNodeData& nd = nb[i];

            if (box.contains(amrex::IntVect(amrex::Dim3{nd.ix, nd.iy, 0}))) {
                if (nd.phi <= 0.0 && nd.fuel_frac > 0.0) {
                    heat_arr(nd.ix, nd.iy, 0) = f_sensible * nd.ir * 1000.0
                                                * nd.fuel_frac * nd.flux_scaling;
                }
            }
        }
    }
}

void FireEngine::GetMoistureFlux(amrex::MultiFab& moisture_flux) const
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::GetMoistureFlux() called before Initialize()");
    }

    // Initialize to zero
    moisture_flux.setVal(0.0);

    // Fraction of total heat released as latent heat (moisture)
    // This includes water vapor from fuel moisture and combustion products
    constexpr amrex::Real f_latent = 0.25;  // 25% latent heat (typical for wildfires)
    constexpr amrex::Real L_v = 2.5e6;      // Latent heat of vaporization (J/kg)

    // Iterate over FABs to handle multi-box decomposition
    for (amrex::MFIter mfi(moisture_flux); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        auto moist_arr = moisture_flux.array(mfi);

        const GpuNarrowBand& nb_m = m_engine->GetNarrowBand();
        int num_nodes_m = nb_m.GetNumNodes();

        for (int i = 0; i < num_nodes_m; i++) {
            const FireNodeData& nd = nb_m[i];

            if (box.contains(amrex::IntVect(amrex::Dim3{nd.ix, nd.iy, 0}))) {
                if (nd.phi <= 0.0 && nd.fuel_frac > 0.0) {
                    amrex::Real latent_heat_flux = f_latent * nd.ir * 1000.0
                                                   * nd.fuel_frac * nd.flux_scaling;
                    moist_arr(nd.ix, nd.iy, 0) = latent_heat_flux / L_v;
                }
            }
        }
    }
}

void FireEngine::GetVelocityField(amrex::MultiFab& vel_u, amrex::MultiFab& vel_v) const
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::GetVelocityField() called before Initialize()");
    }

    // Get velocity from internal engine (it fills the MultiFabs directly)
    m_engine->GetVelocityField(vel_u, vel_v);
}

amrex::Real FireEngine::GetFireArea() const
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::GetFireArea() called before Initialize()");
    }

    // Count cells where phi < 0
    const amrex::MultiFab& phi = m_engine->GetPhi();
    const amrex::Real* dx = m_geom.CellSize();
    amrex::Real cell_area = dx[0] * dx[1];

    amrex::Real local_burned = 0.0;
    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto& phi_arr = phi.const_array(mfi);

        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                if (phi_arr(i, j, 0) < 0.0) {
                    local_burned += cell_area;
                }
            }
        }
    }

    // Sum across all MPI ranks
    amrex::Real total_burned = local_burned;
    amrex::ParallelDescriptor::ReduceRealSum(total_burned);

    return total_burned;
}

int FireEngine::GetNarrowBandSize() const
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::GetNarrowBandSize() called before Initialize()");
    }

    int local_count = m_engine->GetNumActiveCells();
    int global_count = local_count;
    amrex::ParallelDescriptor::ReduceIntSum(global_count);
    return global_count;
}

std::vector<std::pair<int,int>> FireEngine::GetNarrowBandCells() const
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::GetNarrowBandCells() called before Initialize()");
    }

    std::vector<std::pair<int,int>> cells;
    const GpuNarrowBand& nb = m_engine->GetNarrowBand();
    int n_cells = nb.GetNumNodes();
    cells.reserve(n_cells);
    for (int i = 0; i < n_cells; i++) {
        cells.emplace_back(nb[i].ix, nb[i].iy);
    }
    return cells;
}

std::vector<std::string> FireEngine::GetPlotVarNames()
{
    return {"fire_phi", "fire_heat_flux", "fire_moisture_flux", "fire_vel_u", "fire_vel_v", "fire_tagged", "fire_fuel_frac",
            "fire_flin", "fire_ws20", "fire_ember_count", "fire_spot_ignition"};
}

void FireEngine::FillPlotVar(const std::string& name, amrex::MultiFab& mf) const
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::FillPlotVar() called before Initialize()");
    }

    // Helper: build a temporary MultiFab that CPU host loops can read/write.
    // In GPU builds the default arena is device-only; use managed memory so
    // the CPU narrow-band loops inside Get*Field functions work without copies.
    auto make_tmp_mf = [&](int ncomp = 1, int ngrow = 0) {
        amrex::MFInfo info;
        info.SetArena(amrex::The_Managed_Arena());
        return amrex::MultiFab(mf.boxArray(), mf.DistributionMap(), ncomp, ngrow, info);
    };

    if (name == "fire_phi") {
        GetPhiField(mf);
    } else if (name == "fire_heat_flux") {
        GetHeatFlux(mf);
    } else if (name == "fire_moisture_flux") {
        GetMoistureFlux(mf);
    } else if (name == "fire_vel_u") {
        auto vel_v = make_tmp_mf();
        GetVelocityField(mf, vel_v);
    } else if (name == "fire_vel_v") {
        auto vel_u = make_tmp_mf();
        GetVelocityField(vel_u, mf);
    } else if (name == "fire_tagged") {
        m_engine->GetTaggedCells(mf);
    } else if (name == "fire_fuel_frac") {
        // Fill fuel fraction from narrow band nodes
        // Default: 1.0 (unburned = full fuel)
        mf.setVal(1.0);

        // Copy phi into a MultiFab with the SAME BoxArray as mf
        // (mf and internal m_phi may have different box decompositions)
        auto phi_copy = make_tmp_mf();
        GetPhiField(phi_copy);

        // BUG FIX (found via AddressSanitizer, 2026-09-24 — pre-existing,
        // unrelated to the static multilevel work; confirmed present even
        // with all Phase 4 code disabled): this loop used to iterate via
        // MFIter mfi(phi_copy) and write into mf.array(mfi), assuming mf
        // and phi_copy share an identical per-box layout because phi_copy
        // was built from mf.boxArray(). That assumption does not always
        // hold in practice (observed under sustained dynamic-AMR
        // regridding) — ASan caught a WRITE landing thousands of bytes
        // past the end of mf's own allocated box. Fix: drive the loop from
        // mf itself, so the write bound is always mf's own valid box,
        // matching the (already-correct) pattern used in the loop just
        // below this one.
        for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            auto arr = mf.array(mfi);
            auto phi_arr = phi_copy.const_array(mfi);

            // Set burned cells (phi < 0) not in band to fuel_frac = 0 (fully consumed)
            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    if (phi_arr(i, j, 0) < 0.0) {
                        arr(i, j, 0) = 0.0;  // Detagged burned cell = fully consumed
                    }
                }
            }
        }

        // Overwrite with actual fuel_frac from narrow band nodes
        for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            auto arr = mf.array(mfi);

            const GpuNarrowBand& nb_ff = m_engine->GetNarrowBand();
            int n_ff = nb_ff.GetNumNodes();
            for (int i = 0; i < n_ff; i++) {
                const FireNodeData& nd = nb_ff[i];
                if (box.contains(amrex::IntVect(amrex::Dim3{nd.ix, nd.iy, 0}))) {
                    arr(nd.ix, nd.iy, 0) = nd.fuel_frac;
                }
            }
        }
    } else if (name == "fire_ember_count" || name == "fire_spot_ignition") {
        // Spotting diagnostics (zero when spotting is off): embers landed per cell (times the
        // sampling factor) since the start, and 1 where a spot fire was ignited.
        const amrex::MultiFab& src = (name == "fire_ember_count") ? m_engine->GetEmberCount()
                                                                  : m_engine->GetSpotIgnition();
        mf.setVal(0.0);
        mf.ParallelCopy(src, 0, 0, 1);
    } else if (name == "fire_flin" || name == "fire_ws20") {
        // Unburned narrow-band nodes (the advancing front): local fireline intensity
        // [kW/m] from the elliptical spread rate, or the 20-ft wind speed [mph] it used.
        // Zero elsewhere. Diagnostics for spotting (ember distance depends on both).
        const bool flin = (name == "fire_flin");
        mf.setVal(0.0);
        const GpuNarrowBand& nb = m_engine->GetNarrowBand();
        const int n = nb.GetNumNodes();
        for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            auto arr = mf.array(mfi);
            for (int i = 0; i < n; i++) {
                const FireNodeData& nd = nb[i];
                if (nd.burned) continue;
                if (box.contains(amrex::IntVect(amrex::Dim3{nd.ix, nd.iy, 0}))) {
                    arr(nd.ix, nd.iy, 0) = flin ? nd.flin_dms_surface : nd.ws20;
                }
            }
        }
    } else {
        amrex::Abort("FireEngine::FillPlotVar(): Unknown variable name: " + name);
    }
}

void FireEngine::SetTime(amrex::Real time)
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::SetTime() called before Initialize()");
    }
    
    // Forward to internal engine
    m_engine->SetTime(time);
}

void FireEngine::RestorePhiFromCheckpoint(const amrex::MultiFab& phi_restart)
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::RestorePhiFromCheckpoint() called before Initialize()");
    }

    m_engine->RestorePhiFromCheckpoint(phi_restart);
}

void FireEngine::SyncPhiFromHierarchy(const amrex::MultiFab& phi_src)
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::SyncPhiFromHierarchy() called before Initialize()");
    }

    m_engine->SyncPhiFromHierarchy(phi_src);
}

void FireEngine::ApplyValidPhiCorrection(const amrex::MultiFab& candidate,
                                          amrex::Real sync_time,
                                          int edge_jlo, int edge_jhi)
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::ApplyValidPhiCorrection() called before Initialize()");
    }
    m_engine->ApplyValidPhiCorrection(candidate, sync_time, edge_jlo, edge_jhi);
}

PhiCorrectionStats FireEngine::GetLastPhiCorrectionStats() const
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::GetLastPhiCorrectionStats() called before Initialize()");
    }
    return m_engine->GetLastPhiCorrectionStats();
}

void FireEngine::FillGhostPhiFromHierarchy(const amrex::MultiFab& phi_src)
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::FillGhostPhiFromHierarchy() called before Initialize()");
    }

    m_engine->FillGhostPhiFromHierarchy(phi_src);
}

void FireEngine::Regrid(const amrex::BoxArray& new_ba,
                        const amrex::DistributionMapping& new_dm)
{
    if (!m_initialized) {
        amrex::Abort("FireEngine::Regrid() called before Initialize()");
    }

    m_engine->Regrid(new_ba, new_dm);
}

} // namespace ELMFIRE
