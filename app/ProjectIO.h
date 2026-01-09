#pragma once

#include <QString>

class ProjectState;

bool SaveProjectJson(const ProjectState& state, const QString& path);
bool LoadProjectJson(ProjectState& state, const QString& path, QString* error);
