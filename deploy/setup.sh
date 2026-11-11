#!/usr/bin/env bash
# Server side of scripts/switch_to_server.ps1. Run as root; safe to re-run.
#   bash /opt/football-shots/deploy/setup.sh          install/update, leave the timer as it is
#   bash /opt/football-shots/deploy/setup.sh start    ...and start the every-2-hours timer
set -euo pipefail
APP=/opt/football-shots

id shots >/dev/null 2>&1 || useradd --system --home-dir "$APP" --shell /usr/sbin/nologin shots
install -d -o shots -g shots "$APP/data" "$APP/models" "$APP/logs"
chown -R shots:shots "$APP"
[ -f "$APP/.env" ] && chmod 600 "$APP/.env"

if [ ! -x "$APP/.venv/bin/python" ]; then
  echo "== Creating the Python environment (first time only, a few minutes)"
  sudo -u shots python3 -m venv "$APP/.venv"
fi
sudo -u shots "$APP/.venv/bin/pip" install -q --disable-pip-version-check -r "$APP/deploy/requirements-server.txt"

cp "$APP/deploy/football-shots.service" "$APP/deploy/football-shots.timer" /etc/systemd/system/
systemctl daemon-reload

if [ "${1:-}" = "start" ]; then
  systemctl enable -q --now football-shots.timer
  echo "== Timer on. Next runs:"
  systemctl list-timers football-shots.timer --no-pager | head -3
fi
