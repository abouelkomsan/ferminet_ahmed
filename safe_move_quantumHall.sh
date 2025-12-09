#!/usr/bin/env bash
set -euo pipefail

# You can override on run:  DST_BASE=/somewhere ./safe_move_minimalChern.sh
DST_BASE="${DST_BASE:-/ceph/submit/data/user/a/ahmed95/torusquantumHall/}"
SRC_BASE="/work/submit/ahmed95"

mkdir -p "$DST_BASE"

# Raw list of DIRECTORIES (relative to $SRC_BASE)
LIST_RAW=$(cat <<'EOF'
torusquantumHall/ferminet_2025_11_22_11:50:35
torusquantumHall/ferminet_2025_11_22_12:16:31
torusquantumHall/ferminet_2025_11_22_15:26:20
torusquantumHall/ferminet_2025_11_22_17:10:06
torusquantumHall/ferminet_2025_11_22_18:20:02
torusquantumHall/ferminet_2025_11_22_22:16:55
torusquantumHall/ferminet_2025_11_22_22:19:59
torusquantumHall/ferminet_2025_11_22_22:34:18
torusquantumHall/ferminet_2025_11_22_23:17:08
torusquantumHall/ferminet_2025_11_22_23:37:35
torusquantumHall/ferminet_2025_11_22_23:46:51
torusquantumHall/ferminet_2025_11_23_10:55:03
torusquantumHall/ferminet_2025_11_23_11:01:13
torusquantumHall/ferminet_2025_11_23_11:43:37
torusquantumHall/ferminet_2025_11_23_11:59:46
torusquantumHall/ferminet_2025_11_23_12:09:08
torusquantumHall/ferminet_2025_11_23_12:13:20
torusquantumHall/ferminet_2025_11_23_12:16:55
torusquantumHall/ferminet_2025_11_23_12:39:14
torusquantumHall/ferminet_2025_11_23_14:15:18
torusquantumHall/ferminet_2025_11_23_14:27:20
torusquantumHall/ferminet_2025_11_23_14:50:22
torusquantumHall/ferminet_2025_11_23_16:22:26
torusquantumHall/ferminet_2025_11_23_16:35:25
torusquantumHall/ferminet_2025_11_23_18:15:17
torusquantumHall/ferminet_2025_11_23_18:16:04
torusquantumHall/ferminet_2025_11_23_18:22:05
torusquantumHall/ferminet_2025_11_23_23:52:16
torusquantumHall/ferminet_2025_11_23_23:57:06
torusquantumHall/ferminet_2025_11_24_00:11:26
torusquantumHall/ferminet_2025_11_24_09:22:08
torusquantumHall/ferminet_2025_11_24_12:05:40
torusquantumHall/ferminet_2025_11_24_13:25:37
torusquantumHall/ferminet_2025_11_24_16:00:09
torusquantumHall/ferminet_2025_11_24_16:19:11
torusquantumHall/ferminet_2025_11_24_18:39:07
torusquantumHall/ferminet_2025_11_24_18:40:01
torusquantumHall/ferminet_2025_11_24_18:50:42
torusquantumHall/ferminet_2025_11_24_19:25:07
torusquantumHall/ferminet_2025_11_24_22:48:52
torusquantumHall/ferminet_2025_11_25_00:21:31
torusquantumHall/ferminet_2025_11_25_00:31:45
torusquantumHall/ferminet_2025_11_25_00:34:16
torusquantumHall/ferminet_2025_11_25_09:28:01
torusquantumHall/ferminet_2025_11_25_10:42:16
torusquantumHall/ferminet_2025_11_25_12:13:12
torusquantumHall/ferminet_2025_11_25_15:37:24
torusquantumHall/ferminet_2025_11_25_23:06:08
torusquantumHall/ferminet_2025_11_25_23:30:49
torusquantumHall/ferminet_2025_11_25_23:54:55
torusquantumHall/ferminet_2025_11_26_09:37:50
torusquantumHall/ferminet_2025_11_26_09:50:35
torusquantumHall/ferminet_2025_11_26_11:21:57
torusquantumHall/ferminet_2025_11_26_18:44:40
torusquantumHall/ferminet_2025_11_26_19:08:19
torusquantumHall/ferminet_2025_11_26_19:58:54
torusquantumHall/ferminet_2025_11_26_20:51:31
torusquantumHall/ferminet_2025_11_26_21:13:10
torusquantumHall/ferminet_2025_11_27_10:19:53
torusquantumHall/ferminet_2025_11_27_10:41:17
torusquantumHall/ferminet_2025_11_27_10:41:27
torusquantumHall/ferminet_2025_11_28_06:07:20
torusquantumHall/ferminet_2025_11_28_10:12:36
torusquantumHall/ferminet_2025_11_29_10:28:51
torusquantumHall/ferminet_2025_11_29_10:47:32
torusquantumHall/ferminet_2025_11_29_10:47:35
torusquantumHall/ferminet_2025_11_29_12:07:48
torusquantumHall/ferminet_2025_11_29_12:08:31
torusquantumHall/ferminet_2025_11_29_17:52:22
torusquantumHall/ferminet_2025_12_01_08:43:33
torusquantumHall/ferminet_2025_12_01_12:57:06
torusquantumHall/ferminet_2025_12_01_14:26:26
torusquantumHall/ferminet_2025_12_02_10:33:35
torusquantumHall/ferminet_2025_12_02_10:38:22
torusquantumHall/ferminet_2025_12_02_12:33:26
torusquantumHall/ferminet_2025_12_02_12:56:00
torusquantumHall/ferminet_2025_12_02_12:56:17
torusquantumHall/ferminet_2025_12_02_12:56:35
torusquantumHall/ferminet_2025_12_02_13:16:09
torusquantumHall/ferminet_2025_12_02_13:36:52
torusquantumHall/ferminet_2025_12_02_15:23:17
torusquantumHall/ferminet_2025_12_02_15:25:48
torusquantumHall/ferminet_2025_12_02_15:39:02
torusquantumHall/ferminet_2025_12_02_16:09:10
torusquantumHall/ferminet_2025_12_02_16:43:29
torusquantumHall/ferminet_2025_12_02_17:11:53
torusquantumHall/ferminet_2025_12_02_17:58:10
torusquantumHall/ferminet_2025_12_03_12:57:32
torusquantumHall/ferminet_2025_12_03_13:40:23
torusquantumHall/ferminet_2025_12_03_14:20:25
torusquantumHall/ferminet_2025_12_03_14:33:14
torusquantumHall/ferminet_2025_12_03_14:38:12
torusquantumHall/ferminet_2025_12_03_16:00:15
torusquantumHall/ferminet_2025_12_03_16:28:59
torusquantumHall/ferminet_2025_12_03_16:48:12
EOF
)

# Clean up: trim whitespace, remove empty lines
LIST_CLEAN=$(printf '%s\n' "$LIST_RAW" \
  | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' \
  | sed '/^$/d')

cd "$SRC_BASE"

echo "Source base:      $SRC_BASE"
echo "Destination base: $DST_BASE"
echo "Directories to move (contents):"
printf '%s\n' "$LIST_CLEAN"
echo "========================================"

# Optional: quick existence check
echo "== Existence check =="
while IFS= read -r d; do
  if [ -d "$d" ]; then
    echo "OK dir:  $SRC_BASE/$d"
  else
    echo "MISSING dir: $SRC_BASE/$d"
  fi
done <<< "$LIST_CLEAN"
echo "========================================"

# Move contents of each directory
while IFS= read -r d; do
  if [ ! -d "$d" ]; then
    echo "Skipping missing directory: $SRC_BASE/$d"
    continue
  fi

  src_dir="$d"
  dst_dir="$DST_BASE/$d"

  echo
  echo "=== Moving contents of: $SRC_BASE/$src_dir -> $dst_dir ==="
  mkdir -p "$dst_dir"

  # Trailing slash on source means "contents of this directory"
  rsync -aHAXv -s --remove-source-files \
    "$src_dir"/ "$dst_dir"/

  echo "Cleaning up empty directories under: $SRC_BASE/$src_dir"
  find "$src_dir" -depth -type d -empty -print -delete || true
done <<< "$LIST_CLEAN"

echo
echo "Done. All files within those directories should now be under:"
echo "  $DST_BASE"