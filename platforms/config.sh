#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${BUILD_TYPE}" ]; then
   BUILD_TYPE="Release"
fi

LIBSERUM_SHA=2fbe97281000a3b12812cf6af2fd2eaf00f8b2a7

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
