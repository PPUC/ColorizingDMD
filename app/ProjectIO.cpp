#include "ProjectIO.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include "ProjectState.h"

namespace {
static QJsonArray toJsonArray(const QStringList& list)
{
    QJsonArray array;
    for (const auto& item : list) {
        array.append(item);
    }
    return array;
}
}

bool SaveProjectJson(const ProjectState& state, const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly)) {
        return false;
    }

    QJsonObject root;
    root.insert("format", "colorizingdmd-project");
    root.insert("version", 1);
    root.insert("frames", toJsonArray(state.frames()));
    root.insert("sprites", toJsonArray(state.sprites()));
    root.insert("images", toJsonArray(state.images()));

    QJsonDocument doc(root);
    file.write(doc.toJson(QJsonDocument::Indented));
    return true;
}

bool LoadProjectJson(ProjectState& state, const QString& path, QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) {
            *error = "Could not open file";
        }
        return false;
    }

    const QByteArray data = file.readAll();
    QJsonParseError parseError{};
    QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);
    if (doc.isNull()) {
        if (error) {
            *error = parseError.errorString();
        }
        return false;
    }

    if (!doc.isObject()) {
        if (error) {
            *error = "Invalid project format";
        }
        return false;
    }

    const QJsonObject root = doc.object();
    if (root.value("format").toString() != "colorizingdmd-project") {
        if (error) {
            *error = "Unsupported project type";
        }
        return false;
    }

    state.newProject();
    state.openProject(path);

    const QJsonArray frames = root.value("frames").toArray();
    for (const auto& value : frames) {
        if (value.isString()) {
            state.addFrame();
        }
    }

    const QJsonArray sprites = root.value("sprites").toArray();
    for (const auto& value : sprites) {
        if (value.isString()) {
            state.addSprite();
        }
    }

    const QJsonArray images = root.value("images").toArray();
    for (const auto& value : images) {
        if (value.isString()) {
            state.addImportedImage(value.toString());
        }
    }

    return true;
}
