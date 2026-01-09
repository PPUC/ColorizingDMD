#pragma once

#include <string>

#include "legacy_project.h"

bool SaveLegacyProject(const std::string& crom_path,
                       const std::string& rp_path,
                       const LegacyProject& project,
                       std::string* error);
