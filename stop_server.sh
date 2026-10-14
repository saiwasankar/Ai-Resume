#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PIDFILE="$ROOT/.rolefit-server.pid"

if [ ! -f "$PIDFILE" ]; then
  echo "No RoleFit server started by start_server.sh was found."
  exit 0
fi

SERVER_PID=$(cat "$PIDFILE")
case "$SERVER_PID" in
  ''|*[!0-9]*) echo "Invalid server PID file; remove $PIDFILE after checking the process." >&2; exit 1 ;;
esac

if kill -0 "$SERVER_PID" 2>/dev/null; then
  kill -TERM "$SERVER_PID"
  echo "Stopped RoleFit server (PID $SERVER_PID)."
else
  echo "RoleFit server process was already stopped."
fi
rm -f "$PIDFILE"
