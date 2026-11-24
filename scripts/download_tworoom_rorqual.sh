#!/bin/bash

# Kept for the original instructions: fetches the TwoRoom dataset. All datasets now go through
# scripts/download_lewm_data_rorqual.sh (see its header); this is the same as
#   bash scripts/download_lewm_data_rorqual.sh tworoom
set -eo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download_lewm_data_rorqual.sh" tworoom
