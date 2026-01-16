set(APP_NAME "PPUC-Serum-Colorizer")
add_executable(${APP_NAME})

set(HANDBOOK_MD ${CMAKE_SOURCE_DIR}/handbook.md)
set(HANDBOOK_MD_COPY ${CMAKE_CURRENT_BINARY_DIR}/handbook.md)
set(HANDBOOK_QRC ${CMAKE_CURRENT_BINARY_DIR}/handbook.qrc)

add_custom_command(
    OUTPUT ${HANDBOOK_MD_COPY}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HANDBOOK_MD} ${HANDBOOK_MD_COPY}
    DEPENDS ${HANDBOOK_MD}
    VERBATIM
)
add_custom_target(handbook_md DEPENDS ${HANDBOOK_MD_COPY})
add_dependencies(${APP_NAME} handbook_md)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/handbook.qrc.in ${HANDBOOK_QRC} @ONLY)
set_source_files_properties(${HANDBOOK_QRC} PROPERTIES GENERATED TRUE)
set_source_files_properties(${HANDBOOK_MD_COPY} PROPERTIES GENERATED TRUE)

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
