#!/usr/bin/env bash
# Start Label Studio for this repo.
#
# Runs in its own uvx environment: installing it into the project venv pulls in
# opencv-python-headless, which breaks cv2 next to ultralytics' opencv-python.
# The document root must be the repo root, because task URLs are
# /data/local-files/?d=<repo-relative path>. Label Studio additionally needs a
# Local Files storage per project covering the images; setup_label_studio.py
# creates it. .envrc exports the same two variables, for starts without this script.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$REPO_ROOT"

exec uvx label-studio start "$@"
