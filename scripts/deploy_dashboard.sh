#!/usr/bin/env bash
# deploy_dashboard.sh — push the static dashboard to surf.nico.studio on DreamHost
# Run from the project root:  bash scripts/deploy_dashboard.sh

set -euo pipefail

REMOTE_USER="dh_gkku5s"
REMOTE_HOST="pdx1-shared-a1-07.dreamhost.com"
REMOTE_DIR="/home/dh_gkku5s/surf.nico.studio"

echo "==> Deploying dashboard to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

rsync -avzL --progress \
  docs/index.html \
  config/spots.json \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo "==> Fixing permissions..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "chmod 644 ${REMOTE_DIR}/index.html ${REMOTE_DIR}/spots.json"

echo ""
echo "==> Done. Visit https://surf.nico.studio to verify."
