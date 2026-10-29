#include "ConfigReader.H"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <AMReX_Print.H>

using namespace amrex;

void ConfigReader::LoadFromFile(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        Print() << "Warning: Could not open config file: " << filename << "\n";
        return;
    }

    std::string current_section = "";
    std::string line;

    while (std::getline(file, line)) {
        // Remove leading/trailing whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Check for section header [SectionName]
        if (line[0] == '[' && line[line.length()-1] == ']') {
            current_section = line.substr(1, line.length()-2);
            continue;
        }

        // Parse key = value
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);

            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            // Remove inline comments
            size_t comment_pos = value.find('#');
            if (comment_pos != std::string::npos) {
                value = value.substr(0, comment_pos);
                value.erase(value.find_last_not_of(" \t") + 1);
            }

            // Store in map
            m_data[MakeKey(current_section, key)] = value;
        }
    }

    file.close();
    Print() << "Loaded configuration from: " << filename << "\n";
}

int ConfigReader::GetInt(const std::string& section, const std::string& key, int default_value) const
{
    std::string full_key = MakeKey(section, key);
    auto it = m_data.find(full_key);
    if (it != m_data.end()) {
        return std::stoi(it->second);
    }
    return default_value;
}

Real ConfigReader::GetReal(const std::string& section, const std::string& key, Real default_value) const
{
    std::string full_key = MakeKey(section, key);
    auto it = m_data.find(full_key);
    if (it != m_data.end()) {
        return std::stod(it->second);
    }
    return default_value;
}

std::string ConfigReader::GetString(const std::string& section, const std::string& key, const std::string& default_value) const
{
    std::string full_key = MakeKey(section, key);
    auto it = m_data.find(full_key);
    if (it != m_data.end()) {
        return it->second;
    }
    return default_value;
}

bool ConfigReader::GetBool(const std::string& section, const std::string& key, bool default_value) const
{
    std::string full_key = MakeKey(section, key);
    auto it = m_data.find(full_key);
    if (it != m_data.end()) {
        std::string value = it->second;
        // Convert to lowercase
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        return (value == "true" || value == "1" || value == "yes");
    }
    return default_value;
}

bool ConfigReader::HasKey(const std::string& section, const std::string& key) const
{
    std::string full_key = MakeKey(section, key);
    return m_data.find(full_key) != m_data.end();
}
