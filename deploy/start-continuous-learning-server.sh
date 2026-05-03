#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/mrteesoft/ai-signalModel/cryptoTrading-model}"
SERVICE_NAME="${SERVICE_NAME:-ai-signal-continuous-learning}"

cd "$APP_DIR"

git pull --ff-only
chmod +x scripts/startContinuousLearning.sh

sudo cp deploy/ai-signal-continuous-learning.service "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"
sudo systemctl status "${SERVICE_NAME}" --no-pager
