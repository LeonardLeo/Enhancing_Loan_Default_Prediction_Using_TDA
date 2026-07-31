#!/usr/bin/env bash
# Fetches the three portfolio datasets that could not be retrieved in the
# sandbox because their hosts were unreachable. Run on an unrestricted machine.
set -euo pipefail
mkdir -p 04_ziemba_repayment 05_us_p2p_lending 06_sba_7a_504

echo "== 04  Ziemba, Anonymized data about loan repayment and borrowers =="
echo "   DOI 10.17632/fr99jcnkxg.1  |  CC BY 4.0  |  91,759 rows x 273 cols"
# Mendeley serves files via a signed redirect; open the landing page and use
# the 'Download all' button, or use the API below.
curl -L -o 04_ziemba_repayment/dataset.zip \
  "https://data.mendeley.com/public-files/datasets/fr99jcnkxg/files/archive" || \
  echo "   -> fall back to https://data.mendeley.com/datasets/fr99jcnkxg/1"

echo "== 05  Nigmonov et al., US P2P lending platform + state-level features =="
echo "   DOI 10.17632/wb3ndt69gf  |  CC BY 4.0  |  2,703,430 rows"
curl -L -o 05_us_p2p_lending/dataset.zip \
  "https://data.mendeley.com/public-files/datasets/wb3ndt69gf/files/archive" || \
  echo "   -> fall back to https://data.mendeley.com/datasets/wb3ndt69gf"

echo "== 06  SBA 7(a) and 504 FOIA data reports =="
echo "   US public domain  |  approvals since FY1991  |  quarterly refresh"
echo "   Primary portal: https://data.sba.gov/dataset/7-a-504-foia"
echo "   Catalogue record: https://catalog.data.gov/dataset/sba-7a-and-504-loan-data-reports"
echo "   NOTE: record the release date and schema version - the briefing requires"
echo "         a frozen, dated snapshot plus a harmonisation table."

echo
echo "After download: sha256sum every file and append to MANIFEST.csv."
