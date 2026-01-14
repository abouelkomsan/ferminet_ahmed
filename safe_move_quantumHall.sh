#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Override on run:
#   DST_BASE=/ceph/submit/data/user/a/ahmed95/torusquantumHall DRY_RUN=1 ./safe_move_minimalChern.sh
DST_BASE="${DST_BASE:-/ceph/submit/data/user/a/ahmed95/torusquantumHall}"
SRC_BASE="${SRC_BASE:-/work/submit/ahmed95}"

# Normalize DST_BASE (strip trailing slashes)
DST_BASE="${DST_BASE%/}"

command -v rsync >/dev/null 2>&1 || { echo "Error: rsync not found in PATH." >&2; exit 1; }
[[ -d "$SRC_BASE" ]] || { echo "Error: SRC_BASE does not exist: $SRC_BASE" >&2; exit 1; }

mkdir -p "$DST_BASE"

# Raw list of DIRECTORIES (relative to $SRC_BASE) — can be space/newline separated
LIST_RAW=$(
  cat <<'EOF'
torusquantumHall/ferminet_2025_12_22_20:46:21 torusquantumHall/ferminet_2025_12_22_23:32:38
EOF
)

# Build array: any whitespace -> one entry per line; drop empties
mapfile -t DIRS < <(
  printf '%s\n' "$LIST_RAW" \
    | tr -s '[:space:]' '\n' \
    | sed '/^$/d'
)

echo "Source base:      $SRC_BASE"
echo "Destination base: $DST_BASE"
echo "Directories to move (contents):"
printf '  %s\n' "${DIRS[@]}"
echo "========================================"

echo "== Existence check =="
missing=0
for d in "${DIRS[@]}"; do
  if [[ -d "$SRC_BASE/$d" ]]; then
    echo "OK dir:      $SRC_BASE/$d"
  else
    echo "MISSING dir: $SRC_BASE/$d"
    missing=1
  fi
done
echo "========================================"

RSYNC_FLAGS=(-aH --protect-args --remove-source-files)
# Optional safer defaults on mixed filesystems:
# RSYNC_FLAGS+=("--no-owner" "--no-group")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RSYNC_FLAGS+=(--dry-run)
  echo "== DRY RUN enabled =="
fi

for d in "${DIRS[@]}"; do
  src_dir="$SRC_BASE/$d"
  [[ -d "$src_dir" ]] || { echo "Skipping missing directory: $src_dir"; continue; }

  # Avoid double torusquantumHall prefix when DST_BASE already ends with it
  if [[ "$DST_BASE" =~ /torusquantumHall$ && "$d" == torusquantumHall/* ]]; then
    rel="${d#torusquantumHall/}"
  else
    rel="$d"
  fi
  dst_dir="$DST_BASE/$rel"

  echo
  echo "=== Moving contents of: $src_dir -> $dst_dir ==="
  mkdir -p "$dst_dir"

  # The trailing slashes mean “contents of directory”
  rsync "${RSYNC_FLAGS[@]}" -- "$src_dir"/ "$dst_dir"/

  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo "Cleaning up empty directories under: $src_dir"
    find "$src_dir" -depth -type d -empty -print -delete || true
  fi
done

echo
echo "Done. Files should now be under:"
echo "  $DST_BASE"