#!/usr/bin/env bash
# Starts the API and the editor, then prints where to open it. Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x backend/.venv/bin/python ]; then
  python3 -m venv backend/.venv
  backend/.venv/bin/pip install -q -r backend/requirements.txt
fi
if [ ! -d frontend/node_modules ]; then
  (cd frontend && npm install)
fi

trap 'kill 0' EXIT
(cd backend && exec .venv/bin/uvicorn aibricks.main:app --host 127.0.0.1 --port 8731) &
(cd frontend && exec npm run dev -- --host 127.0.0.1) &
wait
