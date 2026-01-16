#!/bin/bash

set -e

PLATFORM="macos"
ARCH="arm64"

source ./platforms/config.sh

BUILD_TYPE=${BUILD_TYPE:-Release}

BUILD_TYPE=${BUILD_TYPE} ./platforms/macos/arm64/external.sh

CMAKE_ARGS=()
if [ -n "${VCPKG_ROOT}" ]; then
   CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}"

mkdir -p "${ARTIFACT_DIR}"
if [ -f "${BUILD_DIR}/app/PPUC-Serum-Colorizer" ]; then
   cp "${BUILD_DIR}/app/PPUC-Serum-Colorizer" "${ARTIFACT_DIR}/"
fi
