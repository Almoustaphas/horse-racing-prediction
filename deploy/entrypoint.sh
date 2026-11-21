#!/bin/bash
# Point d'entrée de l'image.
# Lancé en root (Render), il donne le disque de données à appuser puis abandonne les droits root.
set -euo pipefail

if [ "$(id -u)" = 0 ]; then
    if [ -n "${DATA_DIR:-}" ]; then
        mkdir -p "$DATA_DIR" "${MODELS_DIR:-$DATA_DIR/models}"
        chown -R appuser:appuser "$DATA_DIR"
    fi
    exec setpriv --reuid=appuser --regid=appuser --init-groups "$@"
fi
exec "$@"
