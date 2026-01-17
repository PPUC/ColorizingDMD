set(APP_NAME "PPUC-Serum-Colorizer")
add_executable(${APP_NAME})

set(APP_ICON_PNG ${CMAKE_SOURCE_DIR}/icons/app-icon.png)
set(APP_ICON_ICNS ${CMAKE_SOURCE_DIR}/icons/app-icon.icns)
set(APP_ICON_ICO ${CMAKE_SOURCE_DIR}/icons/app-icon.ico)

set(HANDBOOK_MD ${CMAKE_SOURCE_DIR}/handbook.md)
set(HANDBOOK_QRC ${CMAKE_CURRENT_BINARY_DIR}/handbook.qrc)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/handbook.qrc.in ${HANDBOOK_QRC} @ONLY)
set_source_files_properties(${HANDBOOK_QRC} PROPERTIES GENERATED TRUE)

target_sources(${APP_NAME} PRIVATE
    main.cpp
    MainWindow.cpp
    MainWindow.h
    CanvasWidget.cpp
    CanvasWidget.h
    GLCanvasWidget.cpp
    GLCanvasWidget.h
    ../libserum/src/serum-editor.cpp
    ProjectState.cpp
    ProjectState.h
    ProjectIO.cpp
    ProjectIO.h
    ImageStore.cpp
    ImageStore.h
    ImageLoader.cpp
    ImageLoader.h
    IndexedImageStore.cpp
    IndexedImageStore.h
    resources.qrc
    ${HANDBOOK_QRC}
)

if (APPLE)
    set_target_properties(${APP_NAME} PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_ICON_FILE "app-icon"
    )
    if (EXISTS ${APP_ICON_ICNS})
        set_source_files_properties(${APP_ICON_ICNS} PROPERTIES
            MACOSX_PACKAGE_LOCATION "Resources"
        )
        target_sources(${APP_NAME} PRIVATE ${APP_ICON_ICNS})
    endif ()
endif ()

if (WIN32)
    set_target_properties(${APP_NAME} PROPERTIES WIN32_EXECUTABLE TRUE)
    if (EXISTS ${APP_ICON_ICO})
        target_sources(${APP_NAME} PRIVATE app_icon.rc)
    endif ()
endif ()

option(COLORIZINGDMD_BUILD_WIN32_APP "Build the legacy Win32 UI sources" OFF)

if (WIN32 AND COLORIZINGDMD_BUILD_WIN32_APP)
    target_sources(${APP_NAME} PRIVATE
        ../ColorizingDMD.cpp
        ../OGL_Immediate_2D.cpp
        ../dmddevice.cpp
        ../PupTester.cpp
        ../ResizeFilters.cpp
        ../serumdll.cpp
        ../serumwin.cpp
        ../crc32.cpp
        ../cdecode.cpp
    )
endif ()

target_link_libraries(${APP_NAME} PRIVATE
    colorizing_core
    Qt6::Widgets
    Qt6::Gui
    Qt6::OpenGL
    Qt6::OpenGLWidgets
)

target_include_directories(${APP_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/libserum/src
)
