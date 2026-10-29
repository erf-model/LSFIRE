#ifndef ELMFIRE_CONFIG_READER_H
#define ELMFIRE_CONFIG_READER_H

#include <string>
#include <map>
#include <AMReX_REAL.H>

/**
 * @brief Simple configuration file reader
 *
 * Reads key-value pairs from a text file in the format:
 * key = value
 *
 * Supports sections: [SectionName]
 * Comments: # comment text
 */
class ConfigReader {
public:
    /**
     * @brief Load configuration from file
     */
    void LoadFromFile(const std::string& filename);

    /**
     * @brief Get integer value
     */
    int GetInt(const std::string& section, const std::string& key, int default_value = 0) const;

    /**
     * @brief Get real value
     */
    amrex::Real GetReal(const std::string& section, const std::string& key, amrex::Real default_value = 0.0) const;

    /**
     * @brief Get string value
     */
    std::string GetString(const std::string& section, const std::string& key, const std::string& default_value = "") const;

    /**
     * @brief Get boolean value
     */
    bool GetBool(const std::string& section, const std::string& key, bool default_value = false) const;

    /**
     * @brief Check if key exists
     */
    bool HasKey(const std::string& section, const std::string& key) const;

private:
    std::map<std::string, std::string> m_data;

    std::string MakeKey(const std::string& section, const std::string& key) const {
        return section + "." + key;
    }
};

#endif
