#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${BUILD_TYPE}" ]; then
   BUILD_TYPE="Release"
fi

LIBSERUM_SHA=f871ab8923460108776595170d0bef8f7a3bc8e4

if [ -z "${PLATFORM}" ]; then
   PLATFORM="unknown"
fi

if [ -z "${ARCH}" ]; then
   ARCH="unknown"
fi

if [ -z "${BUILD_DIR}" ]; then
   BUILD_DIR="${ROOT_DIR}/build/${PLATFORM}-${ARCH}"
fi

if [ -z "${ARTIFACT_DIR}" ]; then
   ARTIFACT_DIR="${ROOT_DIR}/artifacts/${PLATFORM}-${ARCH}"
fi

echo "Build type: ${BUILD_TYPE}"
echo "Build dir: ${BUILD_DIR}"
echo "Artifacts: ${ARTIFACT_DIR}"
echo ""
