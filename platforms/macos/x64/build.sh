#!/bin/bash

set -e

PLATFORM="macos"
ARCH="x64"

source ./platforms/config.sh

BUILD_TYPE=${BUILD_TYPE:-Release}

BUILD_TYPE=${BUILD_TYPE} ./platforms/macos/x64/external.sh

CMAKE_ARGS=()
if [ -n "${VCPKG_ROOT}" ]; then
   CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
fi
if [ -n "${CMAKE_PREFIX_PATH}" ]; then
   CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
   if [ -z "${Qt6_DIR}" ]; then
      CMAKE_ARGS+=("-DQt6_DIR=${CMAKE_PREFIX_PATH}/lib/cmake/Qt6")
   fi
elif command -v brew >/dev/null 2>&1; then
   QT_PREFIX=$(brew --prefix qt 2>/dev/null || true)
   if [ -n "${QT_PREFIX}" ]; then
      CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=${QT_PREFIX}")
      CMAKE_ARGS+=("-DQt6_DIR=${QT_PREFIX}/lib/cmake/Qt6")
   fi
fi
if [ -n "${OpenCV_DIR}" ]; then
   CMAKE_ARGS+=("-DOpenCV_DIR=${OpenCV_DIR}")
elif command -v brew >/dev/null 2>&1; then
   OPENCV_PREFIX=$(brew --prefix opencv 2>/dev/null || true)
   if [ -n "${OPENCV_PREFIX}" ]; then
      CMAKE_ARGS+=("-DOpenCV_DIR=${OPENCV_PREFIX}/lib/cmake/opencv4")
   fi
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}"

mkdir -p "${ARTIFACT_DIR}"
if [ -f "${BUILD_DIR}/app/PPUC-Serum-Colorizer" ]; then
   cp "${BUILD_DIR}/app/PPUC-Serum-Colorizer" "${ARTIFACT_DIR}/"
fi
