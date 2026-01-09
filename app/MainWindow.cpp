#include "MainWindow.h"

#include <QAction>
#include <QApplication>
#include <QComboBox>
#include <QDockWidget>
#include <QLabel>
#include <QFile>
#include <QFileDialog>
#include <QFormLayout>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QSpinBox>
#include <QStatusBar>
#include <QStyle>
#include <QTabWidget>
#include <QToolBar>
#include <QSize>
#include <QVBoxLayout>
#include <QWidget>
#include <QFileInfo>
#include <QSignalBlocker>

#include <algorithm>
#include <cctype>

#include <opencv2/imgproc.hpp>

#include "CanvasWidget.h"
#include "ProjectState.h"
#include "ProjectIO.h"
#include "ImageStore.h"
#include "ImageLoader.h"
#include "IndexedImageStore.h"
#include "legacy_project.h"

namespace {
constexpr int kDefaultFrameWidth = 128;
constexpr int kDefaultFrameHeight = 32;
constexpr int kDefaultSpriteWidth = 64;
constexpr int kDefaultSpriteHeight = 64;

cv::Mat EnsureBgr(const cv::Mat& source)
{
    if (source.empty()) {
        return cv::Mat();
    }
    cv::Mat result;
    if (source.channels() == 1) {
        cv::cvtColor(source, result, cv::COLOR_GRAY2BGR);
    } else if (source.channels() == 4) {
        cv::cvtColor(source, result, cv::COLOR_BGRA2BGR);
    } else {
        result = source.clone();
    }
    return result;
}

cv::Mat MakePlaceholderImage(int width, int height, const cv::Scalar& baseColor, const cv::Scalar& gridColor)
{
    cv::Mat image(height, width, CV_8UC3, baseColor);
    const int grid = 8;
    for (int y = grid; y < height; y += grid) {
        cv::line(image, cv::Point(0, y), cv::Point(width, y), gridColor, 1);
    }
    for (int x = grid; x < width; x += grid) {
        cv::line(image, cv::Point(x, 0), cv::Point(x, height), gridColor, 1);
    }
    cv::rectangle(image, cv::Rect(0, 0, width - 1, height - 1), gridColor, 1);
    return image;
}

const cv::Mat* ResolveSourceImage(const QListWidget* imageList,
                                  const ProjectState* state,
                                  const ImageStore* store)
{
    const QListWidgetItem* currentItem = imageList ? imageList->currentItem() : nullptr;
    if (currentItem && state && state->images().contains(currentItem->text())) {
        if (const auto* entry = store->findByPath(currentItem->text())) {
            return &entry->image;
        }
    }
    if (const auto* latest = store->latest()) {
        return &latest->image;
    }
    return nullptr;
}

cv::Mat MakeFittedImage(const cv::Mat& source, int width, int height)
{
    cv::Mat bgr = EnsureBgr(source);
    if (bgr.empty() || width <= 0 || height <= 0) {
        return cv::Mat();
    }
    cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    const double scale = std::min(static_cast<double>(width) / bgr.cols,
                                  static_cast<double>(height) / bgr.rows);
    const int targetW = std::max(1, static_cast<int>(bgr.cols * scale));
    const int targetH = std::max(1, static_cast<int>(bgr.rows * scale));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(targetW, targetH), 0, 0, cv::INTER_AREA);
    const int offsetX = (width - targetW) / 2;
    const int offsetY = (height - targetH) / 2;
    resized.copyTo(output(cv::Rect(offsetX, offsetY, targetW, targetH)));
    return output;
}

bool IsLikelyJsonFile(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }
    while (!file.atEnd()) {
        const char ch = static_cast<char>(file.read(1)[0]);
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            return ch == '{' || ch == '[';
        }
    }
    return false;
}
}

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , m_state(new ProjectState(this))
    , m_imageStore(new ImageStore())
    , m_frameStore(new IndexedImageStore())
    , m_spriteStore(new IndexedImageStore())
{
    setWindowTitle("ColorizingDMD");
    setWindowIcon(QIcon(":/app/app.ico"));
    setMinimumSize(1200, 800);

    auto* fileMenu = menuBar()->addMenu("&File");
    auto* editMenu = menuBar()->addMenu("&Edit");
    auto* viewMenu = menuBar()->addMenu("&View");
    auto* helpMenu = menuBar()->addMenu("&Help");

    auto* newAction = new QAction(QIcon(":/icons/new.png"), "&New", this);
    auto* openAction = new QAction(QIcon(":/icons/open.png"), "&Open...", this);
    auto* saveAction = new QAction(QIcon(":/icons/save.png"), "&Save", this);
    auto* saveAsAction = new QAction("Save &As...", this);
    auto* importImageAction = new QAction(QIcon(":/icons/import.png"), "Import &Image...", this);
    auto* exitAction = new QAction("E&xit", this);
    auto* addFrameAction = new QAction(QIcon(":/icons/add.png"), "Add &Frame", this);
    auto* addSpriteAction = new QAction(QIcon(":/icons/addspr.png"), "Add &Sprite", this);
    auto* removeSelectedAction = new QAction(QIcon(":/icons/remove.png"), "&Remove Selected", this);
    removeSelectedAction->setShortcut(QKeySequence::Delete);
    removeSelectedAction->setShortcutContext(Qt::WidgetWithChildrenShortcut);

    fileMenu->addAction(newAction);
    fileMenu->addAction(openAction);
    m_recentMenu = fileMenu->addMenu("Open &Recent");
    fileMenu->addAction(saveAction);
    fileMenu->addAction(saveAsAction);
    fileMenu->addSeparator();
    fileMenu->addAction(importImageAction);
    fileMenu->addSeparator();
    fileMenu->addAction(exitAction);

    editMenu->addAction(addFrameAction);
    editMenu->addAction(addSpriteAction);
    editMenu->addSeparator();
    editMenu->addAction(removeSelectedAction);

    auto* toolbar = addToolBar("Main");
    toolbar->setIconSize(QSize(22, 22));
    toolbar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    toolbar->addAction(newAction);
    toolbar->addAction(openAction);
    toolbar->addAction(saveAction);
    toolbar->addSeparator();
    toolbar->addAction(importImageAction);
    toolbar->addSeparator();
    toolbar->addAction(addFrameAction);
    toolbar->addAction(addSpriteAction);

    auto* status = statusBar();
    status->showMessage("Qt port: UI scaffolding in progress");

    connect(exitAction, &QAction::triggered, qApp, &QApplication::quit);
    connect(newAction, &QAction::triggered, this, [this]() {
        m_state->newProject();
        m_imageStore->clear();
        m_frameStore->clear();
        m_spriteStore->clear();
        populateBookmarks({}, {});
        statusBar()->showMessage("New project (stub)", 3000);
    });
    connect(openAction, &QAction::triggered, this, [this]() {
        const QString filename = QFileDialog::getOpenFileName(this, "Open Project", QString(), "Serum Projects (*.crom *.cROM *.crp *.cRP);;All Files (*.*)");
        if (!filename.isEmpty()) {
            const QFileInfo info(filename);
            const QString suffix = info.suffix().toLower();
            if (suffix == "crom" || suffix == "crp") {
                if (suffix == "crom" && IsLikelyJsonFile(filename)) {
                    QString error;
                    m_imageStore->clear();
                    m_frameStore->clear();
                    m_spriteStore->clear();
                    populateBookmarks({}, {});
                    if (LoadProjectJson(*m_state, filename, &error)) {
                        statusBar()->showMessage(QString("Open: %1").arg(filename), 5000);
                    } else {
                        statusBar()->showMessage(QString("Open failed: %1").arg(error), 5000);
                    }
                    return;
                }
                QString cromPath = filename;
                QString rpPath;
                if (suffix == "crp") {
                    const QString base = info.completeBaseName();
                    const QString dir = info.absolutePath();
                    const QString candidate = dir + "/" + base + ".crom";
                    if (QFileInfo::exists(candidate)) {
                        cromPath = candidate;
                    } else {
                        const QString candidateUpper = dir + "/" + base + ".cROM";
                        if (QFileInfo::exists(candidateUpper)) {
                            cromPath = candidateUpper;
                        }
                    }
                    rpPath = filename;
                } else {
                    const QString base = info.completeBaseName();
                    const QString dir = info.absolutePath();
                    const QString candidate = dir + "/" + base + ".cRP";
                    if (QFileInfo::exists(candidate)) {
                        rpPath = candidate;
                    }
                }
                LegacyProject legacy;
                std::string error;
                if (!LoadLegacyProject(cromPath.toStdString(), rpPath.toStdString(), legacy, &error)) {
                    statusBar()->showMessage(QString("Open failed: %1").arg(QString::fromStdString(error)), 5000);
                    return;
                }

                m_imageStore->clear();
                m_frameStore->clear();
                m_spriteStore->clear();
                for (const auto& frame : legacy.frames) {
                    m_frameStore->add(frame);
                }
                for (const auto& sprite : legacy.sprites) {
                    m_spriteStore->add(sprite);
                }

                QStringList frames;
                for (int i = 0; i < static_cast<int>(legacy.frames.size()); ++i) {
                    frames.append(QString("Frame %1").arg(i));
                }
                QStringList sprites;
                if (!legacy.sprite_labels.empty()) {
                    for (const auto& label : legacy.sprite_labels) {
                        sprites.append(QString::fromStdString(label));
                    }
                } else {
                    for (int i = 0; i < static_cast<int>(legacy.sprites.size()); ++i) {
                        sprites.append(QString("Sprite %1").arg(i));
                    }
                }

                {
                    QSignalBlocker blocker(m_state);
                    m_state->newProject();
                    m_state->openProject(cromPath);
                    m_state->setFramesAndSprites(frames, sprites);
                }
                updateWindowTitle();
                m_projectLabel->setText(cromPath);
                refreshRecentMenu();
                refreshImageList();
                refreshCounts();
                refreshFrameSpriteLists();
                populateBookmarks(legacy.section_firsts, legacy.section_names);
                statusBar()->showMessage(QString("Open legacy: %1").arg(cromPath), 5000);
                return;
            }
            QString error;
            m_imageStore->clear();
            m_frameStore->clear();
            m_spriteStore->clear();
            populateBookmarks({}, {});
            if (LoadProjectJson(*m_state, filename, &error)) {
                statusBar()->showMessage(QString("Open: %1").arg(filename), 5000);
            } else {
                statusBar()->showMessage(QString("Open failed: %1").arg(error), 5000);
            }
        }
    });
    connect(saveAction, &QAction::triggered, this, [this]() {
        if (m_state->projectPath().isEmpty()) {
            const QString filename = QFileDialog::getSaveFileName(this, "Save Project", QString(), "Serum Projects (*.crom *.cROM *.crp *.cRP)");
            if (filename.isEmpty()) {
                return;
            }
            m_state->saveProject(filename);
            SaveProjectJson(*m_state, filename);
        }
        statusBar()->showMessage(QString("Save: %1").arg(m_state->projectPath()), 5000);
    });
    connect(saveAsAction, &QAction::triggered, this, [this]() {
        const QString filename = QFileDialog::getSaveFileName(this, "Save Project As", QString(), "Serum Projects (*.crom *.cROM *.crp *.cRP)");
        if (!filename.isEmpty()) {
            m_state->saveProject(filename);
            if (SaveProjectJson(*m_state, filename)) {
                statusBar()->showMessage(QString("Save As: %1").arg(filename), 5000);
            } else {
                statusBar()->showMessage("Save failed", 5000);
            }
        }
    });
    connect(importImageAction, &QAction::triggered, this, [this]() {
        const QString filename = QFileDialog::getOpenFileName(this, "Import Image", QString(), "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)");
        if (!filename.isEmpty()) {
            cv::Mat image;
            if (!LoadImageFile(filename, image)) {
                statusBar()->showMessage(QString("Import failed: %1").arg(filename), 5000);
                return;
            }
            m_imageStore->addImage(filename, image);
            m_state->addImportedImage(filename);
            m_imagesCanvas->setTitle(QString("Imported: %1").arg(filename));
            m_imagesCanvas->setStatusText(QString("Imported %1").arg(filename));
            m_imagesCanvas->setImage(image);
            statusBar()->showMessage(QString("Import image: %1").arg(filename), 5000);
        }
    });
    connect(addFrameAction, &QAction::triggered, this, [this]() {
        cv::Mat frameImage;
        const cv::Mat* sourceImage = ResolveSourceImage(m_imagesList, m_state, m_imageStore);
        if (sourceImage && !sourceImage->empty()) {
            frameImage = MakeFittedImage(*sourceImage, kDefaultFrameWidth, kDefaultFrameHeight);
        }
        if (frameImage.empty()) {
            frameImage = MakePlaceholderImage(kDefaultFrameWidth, kDefaultFrameHeight,
                                              cv::Scalar(18, 18, 18),
                                              cv::Scalar(55, 55, 55));
        }
        m_frameStore->add(frameImage);
        m_state->addFrame();
        if (m_framesList->count() > 0) {
            m_framesList->setCurrentRow(m_framesList->count() - 1);
        }
    });
    connect(addSpriteAction, &QAction::triggered, this, [this]() {
        cv::Mat spriteImage;
        const cv::Mat* sourceImage = ResolveSourceImage(m_imagesList, m_state, m_imageStore);
        if (sourceImage && !sourceImage->empty()) {
            spriteImage = MakeFittedImage(*sourceImage, kDefaultSpriteWidth, kDefaultSpriteHeight);
        }
        if (spriteImage.empty()) {
            spriteImage = MakePlaceholderImage(kDefaultSpriteWidth, kDefaultSpriteHeight,
                                               cv::Scalar(24, 24, 24),
                                               cv::Scalar(70, 70, 70));
        }
        m_spriteStore->add(spriteImage);
        m_state->addSprite();
        if (m_spritesList->count() > 0) {
            m_spritesList->setCurrentRow(m_spritesList->count() - 1);
        }
    });
    connect(removeSelectedAction, &QAction::triggered, this, [this]() {
        if (m_framesList->hasFocus()) {
            const int row = m_framesList->currentRow();
            m_frameStore->removeAt(row);
            m_state->removeFrame(row);
        } else if (m_spritesList->hasFocus()) {
            const int row = m_spritesList->currentRow();
            m_spriteStore->removeAt(row);
            m_state->removeSprite(row);
        } else if (m_imagesList->hasFocus()) {
            const int row = m_imagesList->currentRow();
            if (row >= 0 && row < m_state->images().size()) {
                const QString path = m_state->images().at(row);
                m_state->removeImage(row);
                m_imageStore->removeByPath(path);
            }
        }
        updateSelectionFromLists();
    });

    auto* tabs = new QTabWidget(this);
    m_framesCanvas = new CanvasWidget("Frames canvas (placeholder)", tabs);
    m_spritesCanvas = new CanvasWidget("Sprites canvas (placeholder)", tabs);
    m_imagesCanvas = new CanvasWidget("Images canvas (placeholder)", tabs);
    tabs->addTab(m_framesCanvas, "Frames");
    tabs->addTab(m_spritesCanvas, "Sprites");
    tabs->addTab(m_imagesCanvas, "Images");
    setCentralWidget(tabs);

    auto* toolDock = new QDockWidget("Tools", this);
    toolDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    auto* toolsTabs = new QTabWidget(toolDock);
    auto* framesTab = new QWidget(toolsTabs);
    auto* spritesTab = new QWidget(toolsTabs);
    auto* imagesTab = new QWidget(toolsTabs);

    m_frameFilter = new QLineEdit(framesTab);
    m_frameFilter->setPlaceholderText("Filter frames...");
    m_framesList = new QListWidget(framesTab);
    auto* framesLayout = new QVBoxLayout(framesTab);
    framesLayout->addWidget(m_frameFilter);
    framesLayout->addWidget(m_framesList, 1);
    framesTab->setLayout(framesLayout);

    m_spriteFilter = new QLineEdit(spritesTab);
    m_spriteFilter->setPlaceholderText("Filter sprites...");
    m_spritesList = new QListWidget(spritesTab);
    auto* spritesLayout = new QVBoxLayout(spritesTab);
    spritesLayout->addWidget(m_spriteFilter);
    spritesLayout->addWidget(m_spritesList, 1);
    spritesTab->setLayout(spritesLayout);

    m_imagesList = new QListWidget(imagesTab);
    auto* imagesLayout = new QVBoxLayout(imagesTab);
    imagesLayout->addWidget(m_imagesList, 1);
    imagesTab->setLayout(imagesLayout);

    toolsTabs->addTab(framesTab, "Frames");
    toolsTabs->addTab(spritesTab, "Sprites");
    toolsTabs->addTab(imagesTab, "Images");
    toolDock->setWidget(toolsTabs);
    addDockWidget(Qt::LeftDockWidgetArea, toolDock);

    auto* inspectorDock = new QDockWidget("Inspector", this);
    inspectorDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    auto* inspectorWidget = new QWidget(inspectorDock);
    auto* inspectorLayout = new QFormLayout(inspectorWidget);
    m_projectLabel = new QLabel("None", inspectorWidget);
    m_bookmarksCombo = new QComboBox(inspectorWidget);
    m_bookmarksCombo->setEnabled(false);
    m_frameJump = new QSpinBox(inspectorWidget);
    m_frameJump->setEnabled(false);
    m_frameJump->setPrefix("Frame ");
    m_countsLabel = new QLabel("Frames: 0, Sprites: 0", inspectorWidget);
    m_selectionLabel = new QLabel("None", inspectorWidget);
    inspectorLayout->addRow("Project", m_projectLabel);
    inspectorLayout->addRow("Bookmarks", m_bookmarksCombo);
    inspectorLayout->addRow("Go to frame", m_frameJump);
    inspectorLayout->addRow("Counts", m_countsLabel);
    inspectorLayout->addRow("Selection", m_selectionLabel);
    inspectorWidget->setLayout(inspectorLayout);
    inspectorDock->setWidget(inspectorWidget);
    addDockWidget(Qt::RightDockWidgetArea, inspectorDock);

    viewMenu->addAction(toolDock->toggleViewAction());
    viewMenu->addAction(inspectorDock->toggleViewAction());

    auto* aboutAction = new QAction("&About", this);
    helpMenu->addAction(aboutAction);
    connect(aboutAction, &QAction::triggered, this, [this]() {
        QMessageBox::information(this, "About ColorizingDMD", "ColorizingDMD Qt port (UI scaffolding).");
    });

    connect(m_state, &ProjectState::projectPathChanged, this, [this](const QString& path) {
        updateWindowTitle();
        m_projectLabel->setText(path.isEmpty() ? "None" : path);
    });
    connect(m_state, &ProjectState::recentFilesChanged, this, [this]() {
        refreshRecentMenu();
    });
    connect(m_state, &ProjectState::imagesChanged, this, [this]() {
        refreshImageList();
    });
    connect(m_state, &ProjectState::framesChanged, this, [this]() {
        refreshFrameSpriteLists();
    });
    connect(m_state, &ProjectState::spritesChanged, this, [this]() {
        refreshFrameSpriteLists();
    });
    connect(m_state, &ProjectState::countsChanged, this, [this]() {
        refreshCounts();
        refreshFrameSpriteLists();
        updateFrameJumpRange();
    });

    connect(m_bookmarksCombo, &QComboBox::currentIndexChanged, this, [this](int index) {
        if (index < 0) {
            return;
        }
        const QVariant value = m_bookmarksCombo->currentData();
        if (!value.isValid()) {
            return;
        }
        const int frameIndex = value.toInt();
        if (frameIndex >= 0 && frameIndex < m_framesList->count()) {
            m_framesList->setCurrentRow(frameIndex);
        }
    });

    connect(m_framesList, &QListWidget::currentTextChanged, this, [this](const QString& text) {
        if (!text.isEmpty()) {
            setInspectorSelection(QString("Frame: %1").arg(text));
            m_framesCanvas->setTitle(QString("Frame canvas - %1").arg(text));
            m_framesCanvas->setStatusText(QString("Selected %1").arg(text));
            showFrameAtIndex(m_framesList->currentRow());
            if (m_framesList->currentRow() >= 0) {
                QSignalBlocker blocker(m_frameJump);
                m_frameJump->setValue(m_framesList->currentRow());
            }
        } else {
            updateSelectionFromLists();
        }
    });
    connect(m_spritesList, &QListWidget::currentTextChanged, this, [this](const QString& text) {
        if (!text.isEmpty()) {
            setInspectorSelection(QString("Sprite: %1").arg(text));
            m_spritesCanvas->setTitle(QString("Sprite canvas - %1").arg(text));
            m_spritesCanvas->setStatusText(QString("Selected %1").arg(text));
            showSpriteAtIndex(m_spritesList->currentRow());
        } else {
            updateSelectionFromLists();
        }
    });
    connect(m_imagesList, &QListWidget::currentTextChanged, this, [this](const QString& text) {
        if (!text.isEmpty()) {
            setInspectorSelection(QString("Image: %1").arg(text));
            m_imagesCanvas->setTitle(QString("Image canvas - %1").arg(text));
            m_imagesCanvas->setStatusText(QString("Selected %1").arg(text));
            if (m_state->images().contains(text)) {
                showImageForPath(text);
            }
        } else {
            updateSelectionFromLists();
        }
    });

    connect(m_frameFilter, &QLineEdit::textChanged, this, [this](const QString& text) {
        applyFrameFilter(text);
    });
    connect(m_spriteFilter, &QLineEdit::textChanged, this, [this](const QString& text) {
        applySpriteFilter(text);
    });
    connect(m_frameJump, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (value >= 0 && value < m_framesList->count()) {
            m_framesList->setCurrentRow(value);
        }
    });

    refreshRecentMenu();
    refreshImageList();
    refreshCounts();
    refreshFrameSpriteLists();
}

void MainWindow::updateWindowTitle()
{
    if (m_state->projectPath().isEmpty()) {
        setWindowTitle("ColorizingDMD");
        return;
    }
    setWindowTitle(QString("ColorizingDMD - %1").arg(m_state->projectPath()));
}

void MainWindow::refreshRecentMenu()
{
    m_recentMenu->clear();
    for (const auto& entry : m_state->recentFiles()) {
        auto* action = new QAction(entry, this);
        connect(action, &QAction::triggered, this, [this, entry]() {
            m_state->openProject(entry);
            statusBar()->showMessage(QString("Open: %1").arg(entry), 5000);
        });
        m_recentMenu->addAction(action);
    }
    if (m_state->recentFiles().isEmpty()) {
        m_recentMenu->addAction("(Empty)")->setEnabled(false);
    }
}

void MainWindow::refreshImageList()
{
    m_imagesList->clear();
    for (const auto& entry : m_state->images()) {
        m_imagesList->addItem(entry);
    }
    if (m_state->images().isEmpty()) {
        m_imagesList->addItem("No images imported");
        m_imagesCanvas->setTitle("Images canvas (placeholder)");
        m_imagesCanvas->setStatusText("No image selected");
        m_imagesCanvas->setImage(cv::Mat());
        return;
    }
    if (m_imagesList->currentRow() < 0 && m_imagesList->count() > 0) {
        m_imagesList->setCurrentRow(0);
        return;
    }
    const QString currentPath = m_imagesList->currentItem() ? m_imagesList->currentItem()->text() : QString();
    if (!currentPath.isEmpty() && m_state->images().contains(currentPath)) {
        showImageForPath(currentPath);
    } else if (const auto* last = m_imageStore->latest()) {
        if (!last->image.empty()) {
            m_imagesCanvas->setImage(last->image);
        }
    }
}

void MainWindow::refreshCounts()
{
    m_countsLabel->setText(QString("Frames: %1, Sprites: %2").arg(m_state->frameCount()).arg(m_state->spriteCount()));
}

void MainWindow::refreshFrameSpriteLists()
{
    m_framesList->clear();
    m_spritesList->clear();

    if (m_state->frames().isEmpty()) {
        m_framesList->addItem("No frames loaded");
        m_framesCanvas->setTitle("Frames canvas (placeholder)");
        m_framesCanvas->setStatusText("No frames loaded");
        m_framesCanvas->setImage(cv::Mat());
        m_frameStore->clear();
    } else {
        while (m_frameStore->count() < m_state->frames().size()) {
            m_frameStore->add(MakePlaceholderImage(kDefaultFrameWidth, kDefaultFrameHeight,
                                                   cv::Scalar(18, 18, 18),
                                                   cv::Scalar(55, 55, 55)));
        }
        while (m_frameStore->count() > m_state->frames().size()) {
            m_frameStore->removeAt(m_frameStore->count() - 1);
        }
        m_framesList->addItems(m_state->frames());
        if (m_framesList->currentRow() < 0 && m_framesList->count() > 0) {
            m_framesList->setCurrentRow(0);
        } else if (m_framesList->currentRow() >= 0) {
            showFrameAtIndex(m_framesList->currentRow());
        }
    }

    if (m_state->sprites().isEmpty()) {
        m_spritesList->addItem("No sprites loaded");
        m_spritesCanvas->setTitle("Sprites canvas (placeholder)");
        m_spritesCanvas->setStatusText("No sprites loaded");
        m_spritesCanvas->setImage(cv::Mat());
        m_spriteStore->clear();
    } else {
        while (m_spriteStore->count() < m_state->sprites().size()) {
            m_spriteStore->add(MakePlaceholderImage(kDefaultSpriteWidth, kDefaultSpriteHeight,
                                                    cv::Scalar(24, 24, 24),
                                                    cv::Scalar(70, 70, 70)));
        }
        while (m_spriteStore->count() > m_state->sprites().size()) {
            m_spriteStore->removeAt(m_spriteStore->count() - 1);
        }
        m_spritesList->addItems(m_state->sprites());
        if (m_spritesList->currentRow() < 0 && m_spritesList->count() > 0) {
            m_spritesList->setCurrentRow(0);
        } else if (m_spritesList->currentRow() >= 0) {
            showSpriteAtIndex(m_spritesList->currentRow());
        }
    }

    updateFrameJumpRange();
}

void MainWindow::setInspectorSelection(const QString& label)
{
    m_selectionLabel->setText(label);
}

void MainWindow::updateSelectionFromLists()
{
    if (m_framesList->currentRow() >= 0) {
        setInspectorSelection(QString("Frame: %1").arg(m_framesList->currentItem()->text()));
        showFrameAtIndex(m_framesList->currentRow());
        return;
    }
    if (m_spritesList->currentRow() >= 0) {
        setInspectorSelection(QString("Sprite: %1").arg(m_spritesList->currentItem()->text()));
        showSpriteAtIndex(m_spritesList->currentRow());
        return;
    }
    if (m_imagesList->currentRow() >= 0) {
        setInspectorSelection(QString("Image: %1").arg(m_imagesList->currentItem()->text()));
        showImageForPath(m_imagesList->currentItem()->text());
        return;
    }
    setInspectorSelection("None");
}

void MainWindow::showImageForPath(const QString& path)
{
    if (path.isEmpty()) {
        return;
    }
    const auto* entry = m_imageStore->findByPath(path);
    if (!entry) {
        cv::Mat image;
        if (LoadImageFile(path, image)) {
            m_imageStore->addImage(path, image);
            entry = m_imageStore->findByPath(path);
        }
    }
    if (entry && !entry->image.empty()) {
        m_imagesCanvas->setImage(entry->image);
    } else {
        m_imagesCanvas->setImage(cv::Mat());
    }
}

void MainWindow::showFrameAtIndex(int index)
{
    const cv::Mat* image = m_frameStore->at(index);
    if (image && !image->empty()) {
        m_framesCanvas->setImage(*image);
    } else {
        m_framesCanvas->setImage(cv::Mat());
    }
}

void MainWindow::showSpriteAtIndex(int index)
{
    const cv::Mat* image = m_spriteStore->at(index);
    if (image && !image->empty()) {
        m_spritesCanvas->setImage(*image);
    } else {
        m_spritesCanvas->setImage(cv::Mat());
    }
}

void MainWindow::populateBookmarks(const std::vector<uint32_t>& frameStarts,
                                   const std::vector<std::string>& names)
{
    m_bookmarksCombo->clear();
    if (frameStarts.empty() || names.empty()) {
        m_bookmarksCombo->setEnabled(false);
        m_bookmarksCombo->addItem("No bookmarks");
        return;
    }

    const std::size_t count = std::min(frameStarts.size(), names.size());
    int added = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (names[i].empty()) {
            continue;
        }
        const QString label = QString::fromStdString(names[i]) +
            QString(" (Frame %1)").arg(frameStarts[i]);
        m_bookmarksCombo->addItem(label, static_cast<int>(frameStarts[i]));
        ++added;
    }
    if (added == 0) {
        m_bookmarksCombo->setEnabled(false);
        m_bookmarksCombo->addItem("No bookmarks");
    } else {
        m_bookmarksCombo->setEnabled(true);
        m_bookmarksCombo->setCurrentIndex(0);
    }
}

void MainWindow::applyFrameFilter(const QString& text)
{
    const QString needle = text.trimmed();
    for (int i = 0; i < m_framesList->count(); ++i) {
        auto* item = m_framesList->item(i);
        const bool isPlaceholder = item->text().startsWith("No frames");
        if (needle.isEmpty() || isPlaceholder) {
            item->setHidden(false);
        } else {
            item->setHidden(!item->text().contains(needle, Qt::CaseInsensitive));
        }
    }
}

void MainWindow::applySpriteFilter(const QString& text)
{
    const QString needle = text.trimmed();
    for (int i = 0; i < m_spritesList->count(); ++i) {
        auto* item = m_spritesList->item(i);
        const bool isPlaceholder = item->text().startsWith("No sprites");
        if (needle.isEmpty() || isPlaceholder) {
            item->setHidden(false);
        } else {
            item->setHidden(!item->text().contains(needle, Qt::CaseInsensitive));
        }
    }
}

void MainWindow::updateFrameJumpRange()
{
    const int count = m_framesList->count();
    if (count <= 0 || (count == 1 && m_framesList->item(0)->text().startsWith("No frames"))) {
        m_frameJump->setEnabled(false);
        m_frameJump->setRange(0, 0);
        m_frameJump->setValue(0);
        return;
    }
    m_frameJump->setEnabled(true);
    m_frameJump->setRange(0, std::max(0, count - 1));
}
