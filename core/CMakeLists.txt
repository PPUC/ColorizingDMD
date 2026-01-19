add_library(colorizing_core STATIC
    color_utils.cpp
    image_copy.cpp
    image_crop.cpp
    image_processing.cpp
    image_resize.cpp
    legacy_project.cpp
    legacy_project_writer.cpp
    ../libserum/src/SerumData.cpp
    ../libserum/src/SceneGenerator.cpp
    ../libserum/third-party/include/miniz/miniz.c
    ../libserum/third-party/include/lz4/lz4.c
    ../libserum/third-party/include/lz4/lz4hc.c
)

set_source_files_properties(
    ../libserum/third-party/include/miniz/miniz.c
    ../libserum/third-party/include/lz4/lz4.c
    ../libserum/third-party/include/lz4/lz4hc.c
    PROPERTIES
    LANGUAGE C
)

target_include_directories(colorizing_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/libserum/src
    ${CMAKE_SOURCE_DIR}/libserum/third-party/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(colorizing_core PUBLIC ${OpenCV_LIBS})
