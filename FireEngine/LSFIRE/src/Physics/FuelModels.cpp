#include "FuelModels.H"
#include "Core/Constants.H"
#include <AMReX_Print.H>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace amrex;
using namespace ElmfireConstants;

// ============================================================================
// PUBLIC METHODS
// ============================================================================

void FuelModelDatabase::LoadFromCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        amrex::Abort("Cannot open fuel model file: " + filename);
    }

    m_fuel_data.clear();

    std::string line;
    // NOTE: fuel_models.csv has NO header line, so don't skip

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string token;
        FuelModelData data;

        // Parse CSV (comma or space delimited)
        int field = 0;
        while (std::getline(iss, token, ',')) {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);

            if (token.empty()) continue;

            switch (field++) {
                case 0: data.id = std::stoi(token); break;
                case 1: data.name = token; break;
                case 2: // dynamic (.TRUE. or .FALSE.)
                    data.dynamic = (token.find("TRUE") != std::string::npos ||
                                   token.find("true") != std::string::npos);
                    break;
                case 3: data.w_1hr = std::stod(token); break;
                case 4: data.w_10hr = std::stod(token); break;
                case 5: data.w_100hr = std::stod(token); break;
                case 6: data.w_lh = std::stod(token); break;
                case 7: data.w_lw = std::stod(token); break;
                case 8: data.sigma_1hr = std::stod(token); break;
                // Columns 9 and 10 in the ELMFIRE CSV are sigma_lh and sigma_lw (live fuel SAV).
                // sigma_10hr (109 ft⁻¹) and sigma_100hr (30 ft⁻¹) are physical constants
                // for all Rothermel fuel models and are NOT stored per-model in the CSV.
                // 9999 is a sentinel meaning "no live fuel in this class".
                case 9:  { Real s = std::stod(token); data.sigma_lh = (s > 5000.0) ? 1500.0 : s; break; }
                case 10: { Real s = std::stod(token); data.sigma_lw = (s > 5000.0) ? 1500.0 : s; break; }
                case 11: data.depth = std::stod(token); break;
                case 12: data.mex_dead = std::stod(token) / 100.0; break;  // Convert % to fraction
                case 13: data.hoc = std::stod(token); break;
                case 14: data.burnout_time = std::stod(token); break;  // Burnout time (seconds)
                default: break;
            }
        }

        data.has_live_fuels = (data.w_lh > 0.0 || data.w_lw > 0.0);
        m_fuel_data[data.id] = data;
    }

    file.close();

    Print() << "Loaded " << m_fuel_data.size() << " fuel models from " << filename << "\n";
}

FuelModelTable FuelModelDatabase::GetFuelModelTable(int fuel_model_id, Real mlh) {
    if (!HasFuelModel(fuel_model_id)) {
        Print() << "WARNING: Fuel model " << fuel_model_id << " not found, using default (0)\n";
        fuel_model_id = 0;
    }

    // Quantize mlh for caching (30-120% in steps of 1%)
    int mlh_index = std::max(30, std::min(120, static_cast<int>(mlh * 100.0)));

    // Check cache
    int cache_key = (fuel_model_id << 16) | mlh_index;
    if (m_table_cache.find(cache_key) != m_table_cache.end()) {
        return m_table_cache[cache_key];
    }

    // Build new table
    const FuelModelData& data = m_fuel_data[fuel_model_id];
    FuelModelTable table = BuildTableFromData(data, mlh);  // mlh is already a fraction

    // Cache it
    m_table_cache[cache_key] = table;

    return table;
}

void FuelModelDatabase::PrintInfo() const {
    Print() << "Fuel Model Database Info:\n";
    Print() << "  Number of models: " << m_fuel_data.size() << "\n";
    Print() << "  Cache size: " << m_table_cache.size() << "\n";
}

// ============================================================================
// PRIVATE METHODS - TABLE BUILDING
// ============================================================================

FuelModelTable FuelModelDatabase::BuildTableFromData(const FuelModelData& data, Real mlh) {
    FuelModelTable table;

    // ========================================
    // STEP 0: Dynamic fuel adjustment (matches ELMFIRE Fortran lines 790-803)
    // ========================================
    // For dynamic fuels (e.g., GR2), herbaceous fuel is split between live and dead
    // based on live herbaceous moisture content (mlh, 30-120% range)
    // CRITICAL: This must be done BEFORE any other calculations!

    FuelModelData adjusted_data = data;  // Make a modifiable copy

    if (data.dynamic) {
        Real LH = mlh * 100.0;  // Convert fraction to percentage (0.30 -> 30.0)

        // Calculate live and dead fractions (Fortran lines 791-792)
        Real LIVEFRAC = std::max(0.0, std::min((LH - 30.0) / (120.0 - 30.0), 1.0));
        Real DEADFRAC = 1.0 - LIVEFRAC;

        // Split herbaceous fuel (Fortran lines 793-794)
        Real w_lh_csv = data.w_lh;  // Original value from CSV
        Real w_dyn = DEADFRAC * w_lh_csv;  // Dead dynamic fuel
        adjusted_data.w_lh = LIVEFRAC * w_lh_csv;  // Live portion

        // Get sigma for dynamic fuel from CSV column 10 (Fortran line 795)
        // For GR2, this is 1800 ft⁻¹. That column is parsed into data.sigma_lh
        // (see LoadFromCSV: columns 9/10 are sigma_lh/sigma_lw, not sigma_10hr/
        // sigma_100hr, which stay at their fixed Rothermel-constant defaults).
        Real sigma_dyn = data.sigma_lh;

        // Recalculate sigma_1hr as weighted average (Fortran line 796)
        Real sigma_1hr_original = data.sigma_1hr;
        Real w_1hr_original = data.w_1hr;

        if ((sigma_1hr_original * w_1hr_original + sigma_dyn * w_dyn) > SMALL_VALUE) {
            adjusted_data.sigma_1hr = (sigma_1hr_original * sigma_1hr_original * w_1hr_original +
                                       sigma_dyn * sigma_dyn * w_dyn) /
                                      (sigma_1hr_original * w_1hr_original + sigma_dyn * w_dyn);
        }

        // Add dead dynamic fuel to 1-hour fuel (Fortran line 797)
        adjusted_data.w_1hr = w_1hr_original + w_dyn;

        // DEBUG: Print dynamic fuel adjustment
        static bool first_dynamic_debug = true;
        if (first_dynamic_debug && data.id == 102) {
            amrex::Print() << "\n=== DYNAMIC FUEL ADJUSTMENT (Fuel " << data.id << ", mlh=" << (mlh*100) << "%) ===\n";
            amrex::Print() << "  LIVEFRAC = " << LIVEFRAC << ", DEADFRAC = " << DEADFRAC << "\n";
            amrex::Print() << "  w_1hr:  " << w_1hr_original << " -> " << adjusted_data.w_1hr
                          << " lb/ft² (" << (adjusted_data.w_1hr/w_1hr_original) << "×)\n";
            amrex::Print() << "  sigma_1hr: " << sigma_1hr_original << " -> " << adjusted_data.sigma_1hr << " ft⁻¹\n";
            amrex::Print() << "  w_lh:   " << w_lh_csv << " -> " << adjusted_data.w_lh << " lb/ft² (dead -> 1-hr)\n";
            first_dynamic_debug = false;
        }
    }

    // ========================================
    // STEP 1: Compute fuel fractions
    // ========================================
    Real w_total = adjusted_data.w_1hr + adjusted_data.w_10hr + adjusted_data.w_100hr +
                   adjusted_data.w_lh + adjusted_data.w_lw;
    if (w_total < SMALL_VALUE) {
        // Non-burnable fuel model
        return table;
    }

    table.f[0] = adjusted_data.w_1hr / w_total;
    table.f[1] = adjusted_data.w_10hr / w_total;
    table.f[2] = adjusted_data.w_100hr / w_total;
    table.f[3] = 0.0;  // Dynamic dead (not used in most models)
    table.f[4] = adjusted_data.w_lh / w_total;
    table.f[5] = adjusted_data.w_lw / w_total;

    table.f_dead = table.f[0] + table.f[1] + table.f[2] + table.f[3];
    table.f_live = table.f[4] + table.f[5];

    // ========================================
    // STEP 2: Moisture of extinction (live)
    // ========================================
    table.mex_dead = adjusted_data.mex_dead;
    table.mex_live = adjusted_data.mex_live;

    // Compute wprimenumer for live fuel moisture adjustment
    // NOTE: Use adjusted sigma_1hr for dynamic fuels!
    Real sigma[4] = {adjusted_data.sigma_1hr, adjusted_data.sigma_10hr,
                     adjusted_data.sigma_100hr, adjusted_data.sigma_1hr};
    Real w_net[4] = {adjusted_data.w_1hr * 0.99, adjusted_data.w_10hr * 0.99,
                     adjusted_data.w_100hr * 0.99, 0.0};

    for (int i = 0; i < 4; i++) {
        table.wprimenumer[i] = w_net[i] * std::exp(-138.0 / sigma[i]);
    }

    // ========================================
    // STEP 3: Packing ratio and related terms
    // ========================================
    ComputePackingRatioTerms(table, adjusted_data);

    // ========================================
    // STEP 4: Reaction intensity pre-computation
    // ========================================
    ComputeReactionTerms(table, adjusted_data);

    // ========================================
    // STEP 5: Wind and slope coefficients
    // ========================================
    ComputeWindSlopeCoefficients(table, adjusted_data);

    return table;
}

void FuelModelDatabase::ComputePackingRatioTerms(FuelModelTable& table, const FuelModelData& data) {
    // Bulk density (lb/ft³)
    table.rhob = (data.w_1hr + data.w_10hr + data.w_100hr + data.w_lh + data.w_lw) / data.depth;

    // Packing ratio
    table.beta = table.rhob / RHO_PARTICLE;

    // Characteristic surface-area-to-volume ratio (weighted by fuel loading)
    // NOTE: For dynamic fuels, data.sigma_1hr has already been adjusted as weighted average
    Real w_total = data.w_1hr + data.w_10hr + data.w_100hr + data.w_lh + data.w_lw;
    if (w_total > SMALL_VALUE) {
        table.sigma_bar = (data.w_1hr * data.sigma_1hr +
                          data.w_10hr * data.sigma_10hr +
                          data.w_100hr * data.sigma_100hr +
                          data.w_lh * data.sigma_lh +
                          data.w_lw * data.sigma_lw) / w_total;
    } else {
        table.sigma_bar = 2000.0;
    }

    // Optimum packing ratio (from Rothermel 1972, Equation 37)
    table.beta_opt = 3.348 * std::pow(table.sigma_bar, -0.8189);

    // Mineral damping (Anderson 1969)
    table.eta_s = 0.174 * std::pow(data.se_dead, -0.19);
}

void FuelModelDatabase::ComputeReactionTerms(FuelModelTable& table, const FuelModelData& data) {
    // Optimum reaction velocity (Rothermel 1972, Equation 36)
    Real gamma_max = std::pow(table.sigma_bar, 1.5) /
                     (495.0 + 0.0594 * std::pow(table.sigma_bar, 1.5));

    // Reaction velocity ratio
    Real A = 133.0 * std::pow(table.sigma_bar, -0.7913);
    Real ratio = table.beta / table.beta_opt;
    Real gamma_prime = gamma_max * std::pow(ratio, A) *
                       std::exp(A * (1.0 - ratio));

    // Propagating flux ratio (Rothermel 1972, Equation 42)
    Real eps = std::exp(-138.0 / table.sigma_bar);
    table.xi = std::exp((0.792 + 0.681 * std::sqrt(table.sigma_bar)) *
                       (table.beta + 0.1)) / (192.0 + 0.2595 * table.sigma_bar);

    // Store propagating flux for later
    for (int i = 0; i < 6; i++) {
        table.feps[i] = table.f[i] * eps;
    }

    // Pre-compute dead fuel reaction term
    Real w_n_dead = (data.w_1hr + data.w_10hr + data.w_100hr) * 0.99;  // 99% available
    table.gp_wnd_emd_es_hoc = gamma_prime * w_n_dead * data.hoc * table.eta_s;

    // Pre-compute live fuel reaction term (if applicable)
    if (data.has_live_fuels) {
        Real w_n_live = (data.w_lh + data.w_lw) * 0.99;
        table.gp_wnl_eml_es_hoc = gamma_prime * w_n_live * data.hoc * table.eta_s;
    } else {
        table.gp_wnl_eml_es_hoc = 0.0;
    }

    // Residence time (Rothermel 1972, Equation 52): tr = 384 / sigma_bar (minutes)
    if (table.sigma_bar > 1.0) {
        table.tr = 384.0 / table.sigma_bar;
    } else {
        table.tr = 0.2;  // Fallback for non-burnable
    }

    // Burnout time (seconds) - read from fuel_models.csv column 14
    table.burnout_time = data.burnout_time;
}

void FuelModelDatabase::ComputeWindSlopeCoefficients(FuelModelTable& table, const FuelModelData& data) {
    // Wind coefficient C (Rothermel 1972, Equation 47)
    Real c_coeff = 7.47 * std::exp(-0.133 * std::pow(table.sigma_bar, 0.55));

    // Wind exponent B (Rothermel 1972, Equation 48)
    Real b_coeff = 0.02526 * std::pow(table.sigma_bar, 0.54);
    table.b_coeff = b_coeff;
    table.b_coeff_inverse = 1.0 / b_coeff;

    // Slope coefficient (Rothermel 1972, Equation 51)
    table.phisterm = 5.275 * std::pow(table.beta, -0.3);

    // CRITICAL: Apply beta ratio adjustment to get PHIWTERM
    // Fortran: FM%PHIWTERM = FM%C_COEFF * (FM%BETA / FM%BETAOP)**(-FM%E_COEFF)
    Real beta_ratio = table.beta / table.beta_opt;
    Real e_coeff = 0.715 * std::exp(-3.59e-4 * table.sigma_bar);
    Real phiwterm = c_coeff * std::pow(beta_ratio, -e_coeff);
    table.phiwterm = phiwterm;

    // Back-calculation coefficient for WSMFEFF from PHIMAG
    // Fortran: FM%WSMFEFF_COEFF = (1. / FM%PHIWTERM) ** FM%B_COEFF_INVERSE
    table.wsmfeff_coeff = std::pow(1.0 / phiwterm, table.b_coeff_inverse);

    // DEBUG: Print for fuel models 1, 4, and 102
    if (data.id == 1 || data.id == 4 || data.id == 102) {
        amrex::Print() << "\nDEBUG Fuel Model " << data.id << " wind coefficients:\n";
        amrex::Print() << "  sigma_bar = " << table.sigma_bar << " (1/ft)\n";
        amrex::Print() << "  beta = " << table.beta << ", beta_opt = " << table.beta_opt << "\n";
        amrex::Print() << "  beta_ratio = " << beta_ratio << "\n";
        amrex::Print() << "  c_coeff = 7.47 * exp(-0.133 * " << table.sigma_bar << "^0.55) = " << c_coeff << "\n";
        amrex::Print() << "  e_coeff = 0.715 * exp(-3.59e-4 * " << table.sigma_bar << ") = " << e_coeff << "\n";
        amrex::Print() << "  PHIWTERM = c_coeff * (beta/beta_opt)^(-e_coeff)\n";
        amrex::Print() << "           = " << c_coeff << " * " << beta_ratio << "^(-" << e_coeff << ")\n";
        amrex::Print() << "           = " << c_coeff << " * " << std::pow(beta_ratio, -e_coeff) << "\n";
        amrex::Print() << "           = " << phiwterm << "\n";
        amrex::Print() << "  WSMFEFF_COEFF = (1/PHIWTERM)^(1/B) = " << table.wsmfeff_coeff << "\n";
    }
}

