#!/usr/bin/env bash
# deploy_dashboard.sh — push the static dashboard to DreamHost.
# Run from the project root:  bash scripts/deploy_dashboard.sh
#
# Host details come from the environment so this repo can stay public without
# publishing the SSH account and server name. Set them in .env (gitignored):
#
#   DEPLOY_USER=your-dreamhost-user
#   DEPLOY_HOST=your-server.dreamhost.com
#   DEPLOY_DIR=/home/your-dreamhost-user/your.domain

set -euo pipefail

# Load .env if present, without clobbering values already exported.
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

: "${DEPLOY_USER:?set DEPLOY_USER in .env or the environment}"
: "${DEPLOY_HOST:?set DEPLOY_HOST in .env or the environment}"
: "${DEPLOY_DIR:?set DEPLOY_DIR in .env or the environment}"

echo "==> Deploying dashboard to ${DEPLOY_USER}@${DEPLOY_HOST}:${DEPLOY_DIR}"

rsync -avzL --progress \
  docs/index.html \
  config/spots.json \
  "${DEPLOY_USER}@${DEPLOY_HOST}:${DEPLOY_DIR}/"

echo "==> Fixing permissions..."
ssh "${DEPLOY_USER}@${DEPLOY_HOST}" "chmod 644 ${DEPLOY_DIR}/index.html ${DEPLOY_DIR}/spots.json"

echo ""
echo "==> Done. Visit your dashboard URL to verify."
