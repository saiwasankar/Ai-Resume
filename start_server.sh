#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV="$ROOT/.venv"
PIDFILE="$ROOT/.rolefit-server.pid"
LOGFILE="$ROOT/server.log"
PORT="${PORT:-8000}"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "RoleFit is already running (PID $(cat "$PIDFILE"))."
  echo "Open http://127.0.0.1:$PORT"
  echo "Stop it with: ./stop_server.sh"
  exit 0
fi

if [ ! -x "$VENV/bin/python" ]; then
  PYTHON="$(command -v python3 || command -v python)"
  "$PYTHON" -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c 'import fastapi, uvicorn, pdfplumber, sklearn, multipart' >/dev/null 2>&1; then
  echo "Installing backend dependencies into .venv..."
  "$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"
fi

cd "$ROOT"
"$VENV/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" --reload >"$LOGFILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PIDFILE"
trap 'kill "$SERVER_PID" 2>/dev/null || true; rm -f "$PIDFILE"' INT TERM EXIT

echo "RoleFit server started at http://127.0.0.1:$PORT"
echo "API docs: http://127.0.0.1:$PORT/docs"
echo "Logs: $LOGFILE"
echo "Stop from another terminal with ./stop_server.sh, or press Ctrl+C here."

while kill -0 "$SERVER_PID" 2>/dev/null; do
  sleep 1
done
wait "$SERVER_PID" || true
