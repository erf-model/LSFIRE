#ifndef ELMFIRE_FUELMODELS_H
#define ELMFIRE_FUELMODELS_H

#include <AMReX_REAL.H>
#include <string>
#include <map>

/**
 * @brief Raw fuel model data (from CSV file)
 * Matches the structure read from fuel_models.csv
 */
struct FuelModelData {
    int id = 0;
    std::string name = "";
    std::string code = "";

    bool dynamic = false;
    bool has_live_fuels = false;

    // Fuel loadings (lb/ft²)
    amrex::Real w_1hr = 0.0;
    amrex::Real w_10hr = 0.0;
    amrex::Real w_100hr = 0.0;
    amrex::Real w_lh = 0.0;    // Live herbaceous
    amrex::Real w_lw = 0.0;    // Live woody

    // Surface-area-to-volume ratios (1/ft)
    amrex::Real sigma_1hr = 2000.0;
    amrex::Real sigma_10hr = 109.0;
    amrex::Real sigma_100hr = 30.0;
    amrex::Real sigma_lh = 1500.0;
    amrex::Real sigma_lw = 1500.0;

    // Other parameters
    amrex::Real depth = 1.0;           // Fuel bed depth (ft)
    amrex::Real mex_dead = 0.30;       // Dead fuel moisture of extinction
    amrex::Real mex_live = 1.50;       // Live fuel moisture of extinction
    amrex::Real hoc = 8000.0;          // Heat of combustion (BTU/lb)
    amrex::Real burnout_time = 600.0;  // Total burnout time (seconds) - from CSV
    amrex::Real st_dead = 0.0555;      // Dead fuel mineral content
    amrex::Real se_dead = 0.010;       // Dead fuel effective mineral content
    amrex::Real st_live = 0.0555;      // Live fuel mineral content
    amrex::Real se_live = 0.010;       // Live fuel effective mineral content
};

/**
 * @brief Processed fuel model table for Rothermel calculations
 * Matches Fortran TYPE FUEL_MODEL_TABLE_TYPE from ELMFIRE_VARS.f90
 * Pre-computed for efficiency
 */
struct FuelModelTable {
    // ========================================
    // FUEL FRACTIONS (matches Fortran F, FEPS)
    // ========================================
    amrex::Real f[6];              // Fuel fraction by class
    amrex::Real feps[6];           // f * eps (moisture damping)
    amrex::Real wprimenumer[4];    // Used for live moisture calculation
    amrex::Real fmex[4];           // f * mex for dead fuels

    // ========================================
    // MOISTURE OF EXTINCTION
    // ========================================
    amrex::Real mex_dead = 0.30;
    amrex::Real mex_live = 1.50;
    amrex::Real r_mprimedenome14sum_mex_dead = 0.0;

    // ========================================
    // FUEL BED PROPERTIES
    // ========================================
    amrex::Real rhob = 0.0;            // Bulk density (lb/ft³)
    amrex::Real sigma_bar = 2000.0;    // Characteristic SAV (1/ft)
    amrex::Real beta = 0.001;          // Packing ratio
    amrex::Real beta_opt = 0.02;       // Optimum packing ratio

    // ========================================
    // REACTION INTENSITY TERMS
    // ========================================
    amrex::Real eta_s = 0.41;          // Mineral damping coefficient
    amrex::Real xi = 0.4;              // Propagating flux ratio
    amrex::Real tr = 0.2;              // Residence time (min)
    amrex::Real burnout_time = 600.0;  // Total burnout time (seconds)

    amrex::Real gp_wnd_emd_es_hoc = 0.0;     // Dead fuel reaction term
    amrex::Real gp_wnl_eml_es_hoc = 0.0;     // Live fuel reaction term

    // ========================================
    // DEAD/LIVE FRACTIONS
    // ========================================
    amrex::Real f_dead = 0.5;
    amrex::Real f_live = 0.5;

    // ========================================
    // WIND/SLOPE COEFFICIENTS (matches Fortran PHIWTERM, PHISTERM, B_COEFF)
    // ========================================
    amrex::Real phiwterm = 0.0;        // Wind coefficient C
    amrex::Real phisterm = 0.0;        // Slope coefficient
    amrex::Real b_coeff = 1.5;         // Wind exponent B
    amrex::Real b_coeff_inverse = 1.0 / 1.5;
    amrex::Real wsmfeff_coeff = 0.0;   // Effective wind coefficient

    // ========================================
    // CONSTRUCTOR
    // ========================================
    FuelModelTable() {
        for (int i = 0; i < 4; i++) {
            wprimenumer[i] = 0.0;
            fmex[i] = 0.0;
        }
        for (int i = 0; i < 6; i++) {
            f[i] = 0.0;
            feps[i] = 0.0;
        }
    }
};

/**
 * @brief Fuel model database manager
 * Handles loading fuel models from CSV and generating lookup tables
 */
class FuelModelDatabase {
public:
    FuelModelDatabase() = default;
    ~FuelModelDatabase() = default;

    /**
     * @brief Load fuel models from CSV file
     * @param filename Path to fuel_models.csv
     */
    void LoadFromCSV(const std::string& filename);

    /**
     * @brief Get processed fuel model table for given conditions
     * @param fuel_model_id Fuel model ID (1-303)
     * @param mlh Live herbaceous moisture (fraction, 0.30-1.20)
     * @return Pre-computed fuel model table
     */
    FuelModelTable GetFuelModelTable(int fuel_model_id, amrex::Real mlh);

    /**
     * @brief Check if fuel model exists
     */
    bool HasFuelModel(int fuel_model_id) const {
        return m_fuel_data.find(fuel_model_id) != m_fuel_data.end();
    }

    /**
     * @brief Get raw fuel model data
     */
    const FuelModelData& GetFuelModelData(int fuel_model_id) const {
        return m_fuel_data.at(fuel_model_id);
    }

    /**
     * @brief Print fuel model database info
     */
    void PrintInfo() const;

private:
    /**
     * @brief Build processed table from raw data
     * This performs all the Rothermel pre-processing
     */
    FuelModelTable BuildTableFromData(const FuelModelData& data, amrex::Real mlh);

    /**
     * @brief Compute packing ratio terms
     */
    void ComputePackingRatioTerms(FuelModelTable& table, const FuelModelData& data);

    /**
     * @brief Compute reaction intensity terms
     */
    void ComputeReactionTerms(FuelModelTable& table, const FuelModelData& data);

    /**
     * @brief Compute wind/slope coefficients
     */
    void ComputeWindSlopeCoefficients(FuelModelTable& table, const FuelModelData& data);

    // ========================================
    // DATA MEMBERS
    // ========================================

    // Raw fuel model data (from CSV)
    std::map<int, FuelModelData> m_fuel_data;

    // Pre-computed tables (cached for efficiency)
    // Key: (fuel_model_id << 16) | (mlh_index)
    std::map<int, FuelModelTable> m_table_cache;
};

#endif
