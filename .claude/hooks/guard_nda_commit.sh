#!/usr/bin/env bash
# PreToolUse hook (matcher: Bash) — bloquea `git commit` si el staging incluye
# material protegido por el NDA de SoccerNet o secretos.
# Exit 2 = bloquear la herramienta y devolver el mensaje a Claude.

set -euo pipefail

INPUT=$(cat)
CMD=$(printf '%s' "$INPUT" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' 2>/dev/null || echo "")

# Solo nos interesa cuando el comando incluye un git commit
case "$CMD" in
  *"git commit"*) ;;
  *) exit 0 ;;
esac

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" || exit 0

STAGED=$(git diff --cached --name-only 2>/dev/null || true)
[ -z "$STAGED" ] && exit 0

VIOLATIONS=""

# 1) Videos y media en cualquier ruta
MEDIA=$(printf '%s\n' "$STAGED" | grep -Ei '\.(mkv|mp4|avi|mov|webm)$' || true)
[ -n "$MEDIA" ] && VIOLATIONS="${VIOLATIONS}Videos (NDA):\n${MEDIA}\n"

# 2) Imágenes bajo data/ (frames extraídos)
FRAMES=$(printf '%s\n' "$STAGED" | grep -Ei '^data/.*\.(jpg|jpeg|png|bmp|npy|npz)$' || true)
[ -n "$FRAMES" ] && VIOLATIONS="${VIOLATIONS}Frames/features bajo data/ (NDA):\n${FRAMES}\n"

# 3) Archivos de entorno / secretos
ENVS=$(printf '%s\n' "$STAGED" | grep -E '(^|/)\.env(\.|$)?' || true)
[ -n "$ENVS" ] && VIOLATIONS="${VIOLATIONS}Archivos .env:\n${ENVS}\n"

# 4) Contenido staged que mencione la password de SoccerNet
if git diff --cached -G 'SOCCERNET_PASSWORD\s*=\s*["'\''A-Za-z0-9]' --name-only 2>/dev/null | grep -qv '^$'; then
  LEAK=$(git diff --cached -G 'SOCCERNET_PASSWORD\s*=\s*["'\''A-Za-z0-9]' --name-only)
  VIOLATIONS="${VIOLATIONS}Posible password hardcodeada (SOCCERNET_PASSWORD=...):\n${LEAK}\n"
fi

if [ -n "$VIOLATIONS" ]; then
  printf 'COMMIT BLOQUEADO por la política de datos del NDA (ver CLAUDE.md):\n%b\nSacá estos archivos del staging (git restore --staged <archivo>) y revisá el .gitignore.\n' "$VIOLATIONS" >&2
  exit 2
fi

exit 0
