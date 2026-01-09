add_library(colorizing_core STATIC
    color_utils.cpp
    image_copy.cpp
    image_crop.cpp
    image_processing.cpp
    image_resize.cpp
    legacy_project.cpp
)

target_include_directories(colorizing_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(colorizing_core PUBLIC ${OpenCV_LIBS})
