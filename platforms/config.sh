#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${BUILD_TYPE}" ]; then
   BUILD_TYPE="Release"
fi

LIBSERUM_SHA=c8f94a5df229888428a4421e42c1d1b640ec98b7

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
