#!/usr/bin/env bash
# Deploya el stack a Elastic Beanstalk sin dejar el docker-compose.yml de dev
# pisado. El EB CLI empaqueta lo que está en el índice de git (`git add`), no
# el working tree sucio, así que:
#   1. Guarda una copia del docker-compose.yml de dev.
#   2. Lo reemplaza temporalmente por docker-compose.prod.yml y lo stagea.
#   3. Corre `eb deploy --staged`.
#   4. SIEMPRE restaura el original (incluso si el deploy falla), vía trap.
#
# Prerequisitos: `eb init` ya corrido en este repo, credenciales AWS válidas,
# imágenes de api/frontend ya pusheadas a ECR (ver README).
#
# Uso: scripts/deploy_eb.sh [nombre-del-entorno]

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ENV_NAME="${1:-}"
BACKUP="docker-compose.yml.bak"

if [ ! -f docker-compose.prod.yml ]; then
  echo "Falta docker-compose.prod.yml en la raíz del repo." >&2
  exit 1
fi

cleanup() {
  if [ -f "$BACKUP" ]; then
    mv "$BACKUP" docker-compose.yml
    git restore --staged docker-compose.yml 2>/dev/null || true
    echo "docker-compose.yml de dev restaurado."
  fi
}
trap cleanup EXIT

cp docker-compose.yml "$BACKUP"
cp docker-compose.prod.yml docker-compose.yml
git add docker-compose.yml

if [ -n "$ENV_NAME" ]; then
  eb deploy "$ENV_NAME" --staged
else
  eb deploy --staged
fi
