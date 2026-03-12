#!/bin/bash
# CashFlow - DigitalOcean Droplet Setup
# Run on a fresh Ubuntu 22.04+ droplet ($4/mo)
#
# Usage:
#   ssh root@YOUR_DROPLET_IP
#   git clone YOUR_REPO /opt/cashflow
#   cd /opt/cashflow && bash deploy/setup.sh

set -e

echo "=== CashFlow Setup ==="

# System deps
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git

# Create app user
if ! id -u cashflow &>/dev/null; then
    useradd -m -s /bin/bash cashflow
fi

# App directory
APP_DIR="/opt/cashflow"
mkdir -p "$APP_DIR/data/logs"
chown -R cashflow:cashflow "$APP_DIR"

# Python venv
sudo -u cashflow python3 -m venv "$APP_DIR/venv"
sudo -u cashflow "$APP_DIR/venv/bin/pip" install --upgrade pip -q
sudo -u cashflow "$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q

# .env file (user fills in credentials)
if [ ! -f "$APP_DIR/.env" ]; then
    cat > "$APP_DIR/.env" <<'ENVEOF'
# Polymarket wallet (needed for live trading only)
# POLYMARKET_PRIVATE_KEY=
# POLYMARKET_API_KEY=

# Optional: Kalshi API for better arbitrage signals
# KALSHI_EMAIL=
# KALSHI_PASSWORD=
ENVEOF
    chown cashflow:cashflow "$APP_DIR/.env"
    chmod 600 "$APP_DIR/.env"
    echo "Created .env file at $APP_DIR/.env - edit with your credentials"
fi

# Install systemd service
cp "$APP_DIR/deploy/cashflow.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable cashflow

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit credentials: nano $APP_DIR/.env"
echo "  2. Start paper trading: systemctl start cashflow"
echo "  3. Check logs: journalctl -u cashflow -f"
echo "  4. Check status: systemctl status cashflow"
echo ""
echo "The bot starts in PAPER mode by default."
echo "To switch to live: edit /etc/systemd/system/cashflow.service"
echo "  and change --capital 50 to --live --capital 50"
