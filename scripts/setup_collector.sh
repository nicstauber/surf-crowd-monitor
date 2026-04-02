#!/usr/bin/env bash
# setup_collector.sh — run this ON DreamHost after SSH to set up the surf collector
# Usage:  bash ~/surf-crowd-monitor/scripts/setup_collector.sh

set -euo pipefail

PROJECT_DIR="$HOME/surf-crowd-monitor"
VENV_DIR="$PROJECT_DIR/venv"
BIN_DIR="$HOME/bin"

echo "============================================"
echo "  Surf Crowd Monitor — DreamHost Setup"
echo "============================================"
echo ""

# ── 1. Check / install ffmpeg ──────────────────────────────────────────────
echo "--> Checking for ffmpeg..."
if command -v ffmpeg &>/dev/null; then
  echo "    ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
  echo "    ffmpeg not found. Downloading static binary..."
  mkdir -p "$BIN_DIR"
  cd "$BIN_DIR"
  FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
  curl -L -o ffmpeg-static.tar.xz "$FFMPEG_URL"
  tar -xf ffmpeg-static.tar.xz --strip-components=1 --wildcards '*/ffmpeg'
  rm ffmpeg-static.tar.xz
  chmod +x "$BIN_DIR/ffmpeg"
  # Persist PATH update
  if ! grep -q 'export PATH="$HOME/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
  fi
  export PATH="$HOME/bin:$PATH"
  echo "    ffmpeg installed: $(ffmpeg -version 2>&1 | head -1)"
fi
echo ""

# ── 2. Python virtualenv ───────────────────────────────────────────────────
echo "--> Setting up Python virtualenv..."
cd "$PROJECT_DIR"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv venv
  echo "    Created venv at $VENV_DIR"
else
  echo "    venv already exists, skipping creation"
fi

source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "    Dependencies installed."
echo ""

# ── 3. .env file ──────────────────────────────────────────────────────────
echo "--> Checking .env file..."
if [ ! -f "$PROJECT_DIR/.env" ]; then
  cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
  echo ""
  echo "    *** ACTION REQUIRED ***"
  echo "    Created .env from template. You must fill in your secrets:"
  echo ""
  echo "      nano $PROJECT_DIR/.env"
  echo ""
  echo "    Required values:"
  echo "      ANTHROPIC_API_KEY   — your Anthropic key (sk-ant-...)"
  echo "      SUPABASE_URL        — https://xxxx.supabase.co"
  echo "      SUPABASE_SERVICE_KEY — service role key (eyJ...)"
  echo ""
  echo "    Re-run this script after filling in .env to complete setup."
  exit 0
else
  echo "    .env exists."
fi

# Verify secrets are not placeholders
source "$PROJECT_DIR/.env" 2>/dev/null || true
if [[ "${ANTHROPIC_API_KEY:-}" == "sk-ant-..." ]] || [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo ""
  echo "    ERROR: ANTHROPIC_API_KEY is not set in .env"
  echo "    Edit $PROJECT_DIR/.env and fill in the real values."
  exit 1
fi
echo ""

# ── 4. Smoke test ─────────────────────────────────────────────────────────
echo "--> Running smoke test (--once --force, single spot)..."
cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"
python src/scheduler.py --once --force
echo ""
echo "    Smoke test complete. Check Supabase for a new observation row."
echo ""

# ── 5. Cron job ───────────────────────────────────────────────────────────
echo "--> Installing cron job (every 15 minutes)..."

CRON_CMD="*/15 * * * * $VENV_DIR/bin/python $PROJECT_DIR/src/scheduler.py --once >> $PROJECT_DIR/cron.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "scheduler.py"; then
  echo "    Cron job already installed:"
  crontab -l | grep "scheduler.py"
else
  # Append to existing crontab
  ( crontab -l 2>/dev/null; echo "$CRON_CMD" ) | crontab -
  echo "    Cron job installed:"
  echo "    $CRON_CMD"
fi
echo ""

echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Monitor logs:  tail -f $PROJECT_DIR/cron.log"
echo "  Test manually: $VENV_DIR/bin/python $PROJECT_DIR/src/scheduler.py --once --force"
echo "============================================"
