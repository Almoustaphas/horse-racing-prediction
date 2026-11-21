#!/bin/bash
# Production sur une seule machine (Render) : API + planificateur dans le même conteneur.
# Si l'un des deux s'arrête, le conteneur s'arrête et la plateforme le redémarre.
set -uo pipefail

python -m api.jobs scheduler &
uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" \
    --proxy-headers --forwarded-allow-ips='*' &

trap 'kill -TERM $(jobs -p) 2>/dev/null' TERM INT
wait -n
status=$?
kill -TERM $(jobs -p) 2>/dev/null || true
wait
exit "$status"
