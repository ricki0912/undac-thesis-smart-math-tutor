#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  configs \
  data/external \
  data/processed \
  data/raw \
  models \
  notebooks \
  reports/figures \
  reports/metrics \
  scripts \
  src/data \
  src/models \
  src/api \
  src/features \
  src/training \
  src/evaluation \
  src/utils

touch \
  configs/.gitkeep \
  data/external/.gitkeep \
  data/processed/.gitkeep \
  data/raw/.gitkeep \
  models/.gitkeep \
  reports/.gitkeep \
  reports/figures/.gitkeep \
  reports/metrics/.gitkeep \
  src/features/.gitkeep \
  src/training/.gitkeep \
  src/evaluation/.gitkeep \
  src/utils/.gitkeep

echo "Estructura creada correctamente."
