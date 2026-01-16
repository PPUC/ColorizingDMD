#!/bin/bash

set -e

source ./platforms/config.sh

if [ ! -d "${ROOT_DIR}/libserum/.git" ]; then
   echo "Fetching libserum (${LIBSERUM_SHA})..."
   rm -rf "${ROOT_DIR}/libserum"
   git clone https://github.com/PPUC/libserum.git "${ROOT_DIR}/libserum"
   (cd "${ROOT_DIR}/libserum" && git checkout "${LIBSERUM_SHA}")
fi

echo "External dependencies are provided by the CI runner."
