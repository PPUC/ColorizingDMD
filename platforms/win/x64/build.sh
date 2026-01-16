#!/bin/bash

set -e

PLATFORM="win"
ARCH="x64"

source ./platforms/config.sh

BUILD_TYPE=${BUILD_TYPE:-Release}

BUILD_TYPE=${BUILD_TYPE} ./platforms/win/x64/external.sh

CMAKE_ARGS=("-G" "Visual Studio 17 2022" "-A" "x64")
if [ -n "${VCPKG_ROOT}" ]; then
   CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}"

mkdir -p "${ARTIFACT_DIR}"
if [ -f "${BUILD_DIR}/app/${BUILD_TYPE}/PPUC-Serum-Colorizer.exe" ]; then
   cp "${BUILD_DIR}/app/${BUILD_TYPE}/PPUC-Serum-Colorizer.exe" "${ARTIFACT_DIR}/"
fi
