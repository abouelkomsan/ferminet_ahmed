#!/usr/bin/env bash
set -euo pipefail

# You can override on run:  DST_BASE=/somewhere ./safe_move_minimalChern.sh
DST_BASE="${DST_BASE:-/ceph/submit/data/user/a/ahmed95/torusquantumHall/}"
SRC_BASE="/work/submit/ahmed95"

mkdir -p "$DST_BASE"

# Raw list of directories (keep as-is here; we'll clean it below)
LIST_RAW=$(cat <<'EOF'
torusquantumHall/ferminet_2025_11_13_08:50:12/
torusquantumHall/ferminet_2025_11_13_12:04:43/
torusquantumHall/ferminet_2025_11_13_12:16:14/ 
torusquantumHall/ferminet_2025_11_13_12:20:18/ 
torusquantumHall/ferminet_2025_11_13_12:38:51/ 
torusquantumHall/ferminet_2025_11_13_12:42:15/ 
torusquantumHall/ferminet_2025_11_13_14:36:15/ 
torusquantumHall/ferminet_2025_11_13_16:00:05/ 
torusquantumHall/ferminet_2025_11_13_16:01:32/ 
torusquantumHall/ferminet_2025_11_13_16:04:47/ 
torusquantumHall/ferminet_2025_11_13_16:21:05/ 
torusquantumHall/ferminet_2025_11_13_16:40:57/ 
torusquantumHall/ferminet_2025_11_13_16:57:48/ 
torusquantumHall/ferminet_2025_11_13_17:32:00/ 
torusquantumHall/ferminet_2025_11_13_19:44:13/ 
torusquantumHall/ferminet_2025_11_13_21:01:39/ 
torusquantumHall/ferminet_2025_11_13_21:31:54/ 
torusquantumHall/ferminet_2025_11_13_21:45:02/ 
torusquantumHall/ferminet_2025_11_13_22:01:58/ 
torusquantumHall/ferminet_2025_11_13_22:05:28/ 
torusquantumHall/ferminet_2025_11_13_23:11:48/ 
torusquantumHall/ferminet_2025_11_13_23:15:50/ 
torusquantumHall/ferminet_2025_11_13_23:35:32/ 
torusquantumHall/ferminet_2025_11_14_09:14:47/ 
torusquantumHall/ferminet_2025_11_14_09:17:50/
torusquantumHall/ferminet_2025_11_14_09:30:18/ 
torusquantumHall/ferminet_2025_11_14_09:30:20/ 
torusquantumHall/ferminet_2025_11_14_09:30:25/ 
torusquantumHall/ferminet_2025_11_14_09:45:44/
torusquantumHall/ferminet_2025_11_14_09:46:29/ 
torusquantumHall/ferminet_2025_11_14_10:05:43/
torusquantumHall/ferminet_2025_11_14_10:05:51/ 
torusquantumHall/ferminet_2025_11_14_10:19:23/ 
torusquantumHall/ferminet_2025_11_14_10:32:19/ 
torusquantumHall/ferminet_2025_11_14_10:34:58/ 
torusquantumHall/ferminet_2025_11_14_11:55:28/ 
torusquantumHall/ferminet_2025_11_14_14:20:02/ 
torusquantumHall/ferminet_2025_11_14_15:16:00/ 
torusquantumHall/ferminet_2025_11_14_15:31:53/ 
torusquantumHall/ferminet_2025_11_14_15:59:21/ 
torusquantumHall/ferminet_2025_11_14_16:03:02/
torusquantumHall/ferminet_2025_11_14_16:55:29/ 
torusquantumHall/ferminet_2025_11_14_17:13:31/ 
torusquantumHall/ferminet_2025_11_14_17:59:24/ 
torusquantumHall/ferminet_2025_11_14_18:27:25/ 
torusquantumHall/ferminet_2025_11_14_20:39:25/ 
torusquantumHall/ferminet_2025_11_14_21:33:16/ 
torusquantumHall/ferminet_2025_11_14_22:58:18/ 
torusquantumHall/ferminet_2025_11_14_23:04:41/ 
torusquantumHall/ferminet_2025_11_14_23:48:22/ 
torusquantumHall/ferminet_2025_11_15_01:39:34/ 
torusquantumHall/ferminet_2025_11_15_16:32:22/ 
torusquantumHall/ferminet_2025_11_15_18:46:07/ 
torusquantumHall/ferminet_2025_11_15_18:52:46/ 
torusquantumHall/ferminet_2025_11_15_19:41:28/ 
torusquantumHall/ferminet_2025_11_15_21:04:23/ 
torusquantumHall/ferminet_2025_11_15_21:04:40/ 
torusquantumHall/ferminet_2025_11_15_21:04:52/ 
torusquantumHall/ferminet_2025_11_15_21:20:11/ 
torusquantumHall/ferminet_2025_11_15_21:32:52/ 
torusquantumHall/ferminet_2025_11_15_23:42:46/ 
torusquantumHall/ferminet_2025_11_16_09:56:25/ 
torusquantumHall/ferminet_2025_11_16_11:01:47/ 
torusquantumHall/ferminet_2025_11_16_11:11:00/ 
torusquantumHall/ferminet_2025_11_16_12:19:46/ 
torusquantumHall/ferminet_2025_11_16_12:33:20/ 
torusquantumHall/ferminet_2025_11_16_18:15:46/ 
torusquantumHall/ferminet_2025_11_16_18:19:43/ 
torusquantumHall/ferminet_2025_11_17_08:22:26/ 
torusquantumHall/ferminet_2025_11_17_08:24:10/ 
torusquantumHall/ferminet_2025_11_17_09:56:21/ 
torusquantumHall/ferminet_2025_11_17_12:09:36/ 
torusquantumHall/ferminet_2025_11_17_13:49:10/ 
torusquantumHall/ferminet_2025_11_17_15:25:59/ 
torusquantumHall/ferminet_2025_11_17_15:57:22/ 
torusquantumHall/ferminet_2025_11_17_18:54:42/ 
torusquantumHall/ferminet_2025_11_17_19:05:23/ 
torusquantumHall/ferminet_2025_11_17_19:27:23/ 
torusquantumHall/ferminet_2025_11_17_19:38:06/ 
torusquantumHall/ferminet_2025_11_17_19:48:07/ 
torusquantumHall/ferminet_2025_11_17_20:01:09/ 
torusquantumHall/ferminet_2025_11_17_20:50:51/ 
torusquantumHall/ferminet_2025_11_17_23:30:26/ 
torusquantumHall/ferminet_2025_11_17_23:34:45/
torusquantumHall/ferminet_2025_11_17_23:48:07/ 
torusquantumHall/ferminet_2025_11_17_23:52:55/ 
torusquantumHall/ferminet_2025_11_18_00:08:32/ 
torusquantumHall/ferminet_2025_11_18_00:12:26/ 
torusquantumHall/ferminet_2025_11_18_00:45:16/ 
torusquantumHall/ferminet_2025_11_18_01:32:44/ 
torusquantumHall/ferminet_2025_11_18_01:46:29/ 
torusquantumHall/ferminet_2025_11_18_01:48:15/ 
torusquantumHall/ferminet_2025_11_18_09:27:58/ 
torusquantumHall/ferminet_2025_11_18_09:41:24/ 
torusquantumHall/ferminet_2025_11_18_10:45:59/ 
torusquantumHall/ferminet_2025_11_18_11:06:46/ 
torusquantumHall/ferminet_2025_11_18_13:05:39/ 
torusquantumHall/ferminet_2025_11_18_14:17:59/ 
torusquantumHall/ferminet_2025_11_18_15:02:21/ 
torusquantumHall/ferminet_2025_11_18_15:03:11/ 
torusquantumHall/ferminet_2025_11_18_15:16:15/ 
torusquantumHall/ferminet_2025_11_18_15:19:55/ 
torusquantumHall/ferminet_2025_11_18_15:32:53/ 
torusquantumHall/ferminet_2025_11_18_15:37:16/ 
torusquantumHall/ferminet_2025_11_18_15:39:29/ 
torusquantumHall/ferminet_2025_11_18_15:57:09/ 
torusquantumHall/ferminet_2025_11_18_16:16:15/ 
torusquantumHall/ferminet_2025_11_18_16:28:28/ 
torusquantumHall/ferminet_2025_11_18_16:31:56/ 
torusquantumHall/ferminet_2025_11_18_17:19:55/ 
torusquantumHall/ferminet_2025_11_18_17:20:08/ 
torusquantumHall/ferminet_2025_11_18_17:40:08/ 
torusquantumHall/ferminet_2025_11_18_18:22:05/ 
torusquantumHall/ferminet_2025_11_18_23:06:20/ 
torusquantumHall/ferminet_2025_11_18_23:08:04/ 
torusquantumHall/ferminet_2025_11_18_23:20:14/ 
torusquantumHall/ferminet_2025_11_18_23:34:05/ 
torusquantumHall/ferminet_2025_11_18_23:48:25/
EOF
)

# Clean up the list:
# - trim leading/trailing whitespace on each line
# - drop completely empty lines
LIST_CLEAN=$(printf '%s\n' "$LIST_RAW" \
  | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' \
  | sed '/^$/d')

cd "$SRC_BASE"

# Convenience: create one process substitution command for all rsync uses
FILES_FROM_CLEAN="<(printf '%s\n' \"$LIST_CLEAN\")"

echo "== Dry-run =="
# shellcheck disable=SC2086
rsync -aHAXnv -s --info=stats1,progress2 \
  --files-from=<(printf '%s\n' "$LIST_CLEAN") \
  ./ "$DST_BASE/"

echo "== Copying =="
# shellcheck disable=SC2086
rsync -aHAX -s --info=stats1,progress2 \
  --files-from=<(printf '%s\n' "$LIST_CLEAN") \
  ./ "$DST_BASE/"

echo "== Verifying by checksum (should print nothing) =="
# shellcheck disable=SC2086
rsync -aHAXcni -s --delete \
  --files-from=<(printf '%s\n' "$LIST_CLEAN") \
  ./ "$DST_BASE/"

echo "== Removing originals safely (only files already copied) =="
# shellcheck disable=SC2086
rsync -aHAX -s --remove-source-files \
  --files-from=<(printf '%s\n' "$LIST_CLEAN") \
  ./ "$DST_BASE/"

echo "== Cleaning up now-empty source directories =="
printf '%s\n' "$LIST_CLEAN" | while IFS= read -r d; do
  # Only try to delete if directory actually exists
  if [ -d "$d" ]; then
    find "$d" -depth -type d -empty -delete || true
  fi
done

echo "Done. Moved into: $DST_BASE"