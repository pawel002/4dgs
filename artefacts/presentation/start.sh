#!/usr/bin/env bash
# Launch the defense presentation.
# The deck uses ES modules, so it must be served over HTTP (not opened as file://).
cd "$(dirname "$0")"
PORT="${1:-8000}"
( sleep 1; open "http://localhost:$PORT" ) &
python3 -m http.server "$PORT" --bind 127.0.0.1
