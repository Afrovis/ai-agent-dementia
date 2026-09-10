#!/usr/bin/env bash
# Downloads the IndoorActionDataset (BSD 3-Clause,
# https://github.com/DaniDeniz/IndoorActionDataset) for tier 2 of the
# perception bench (issue #11), and unpacks it into
# data/perception_bench/daylight/.
#
# Opt-in and never run automatically: this is a ~977 MB download. Nothing
# it fetches is ever committed -- data/ is gitignored -- and this script is
# the only thing in this bench that touches the network.
#
# Usage: tests/perception_bench/fetch_daylight.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/perception_bench"
DAYLIGHT_DIR="${DATA_DIR}/daylight"
ARCHIVE="${DATA_DIR}/indoor_action_dataset.zip"
URL="https://atcdatos.ugr.es/index.php/s/zqPA9ajR78Bn6qB/download"

mkdir -p "${DATA_DIR}"

if [ -d "${DAYLIGHT_DIR}" ] && [ -n "$(ls -A "${DAYLIGHT_DIR}" 2>/dev/null)" ]; then
    echo "Already populated: ${DAYLIGHT_DIR}"
    echo "Delete it first if you want to re-download."
    exit 0
fi

echo "Downloading IndoorActionDataset (~977 MB) to ${ARCHIVE}..."
curl -L --fail --progress-bar -o "${ARCHIVE}" "${URL}"

echo "Unpacking to ${DAYLIGHT_DIR}..."
mkdir -p "${DAYLIGHT_DIR}"
unzip -q "${ARCHIVE}" -d "${DAYLIGHT_DIR}"
rm -f "${ARCHIVE}"

echo "Done. Inspect ${DAYLIGHT_DIR} and, if the class directory names don't"
echo "match perception_bench/daylight.py's _CLASS_KEYWORDS, update that"
echo "table rather than renaming the extracted data."
