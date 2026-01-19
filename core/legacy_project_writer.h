#pragma once

#include <string>

#include "legacy_project.h"

class SerumData;

bool SaveLegacyProject(const std::string& crom_path,
                       const std::string& rp_path,
                       const LegacyProject& project,
                       std::string* error);

bool SaveLegacyProjectRp(const std::string& rp_path,
                         const LegacyProject& project,
                         std::string* error);

bool SaveConcentrateProject(const std::string& cromc_path,
                            const LegacyProject& project,
                            std::string* error);

bool BuildConcentrateData(const LegacyProject& project,
                          SerumData& out,
                          std::string* error);
