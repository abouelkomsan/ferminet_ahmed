#!/usr/bin/env bash
set -euo pipefail

# You can override on run:
#   DST_BASE=/somewhere ./safe_move_minimalChern.sh
DST_BASE="${DST_BASE:-/ceph/submit/data/user/a/ahmed95/minimalChern_NN/}"

# IMPORTANT: point directly to the minimalChern_NN directory
SRC_BASE="/work/submit/ahmed95/minimalChern_NN"

mkdir -p "$DST_BASE"

cd "$SRC_BASE"

# List of run directories (now WITHOUT the "minimalChern_NN/" prefix)
FILELIST=$(cat <<'EOF'
ferminet_2025_10_18_10:11:32
ferminet_2025_10_18_13:11:19
ferminet_2025_10_18_14:08:20
ferminet_2025_10_18_16:25:10
ferminet_2025_10_18_21:08:51
ferminet_2025_10_20_10:01:49
ferminet_2025_10_20_17:14:44
ferminet_2025_10_20_17:15:28
ferminet_2025_10_20_17:15:46
ferminet_2025_10_20_17:26:11
ferminet_2025_10_20_17:48:20
ferminet_2025_10_21_13:29:32
ferminet_2025_10_21_13:35:22
ferminet_2025_10_21_13:37:17
ferminet_2025_10_21_13:42:59
ferminet_2025_10_25_15:30:44
ferminet_2025_10_26_19:25:04
ferminet_2025_10_27_08:40:02
ferminet_2025_10_27_09:46:05
ferminet_2025_10_27_10:01:26
ferminet_2025_10_27_19:51:14
ferminet_2025_10_29_15:09:49
ferminet_2025_10_29_18:36:49
ferminet_2025_10_31_14:20:51
ferminet_2025_10_31_19:36:46
ferminet_2025_10_31_19:41:04
ferminet_2025_10_31_19:49:12
ferminet_2025_10_31_20:03:50
ferminet_2025_11_07_13:59:12
ferminet_2025_11_07_14:08:40
ferminet_2025_11_07_18:25:30
ferminet_2025_11_07_19:03:50
ferminet_2025_11_07_19:06:16
ferminet_2025_11_07_19:19:23
ferminet_2025_11_07_19:33:14
ferminet_2025_11_07_19:49:54
ferminet_2025_11_08_20:40:17
ferminet_2025_11_08_20:42:35
ferminet_2025_11_08_21:26:20
ferminet_2025_11_08_21:27:20
ferminet_2025_11_08_21:27:54
ferminet_2025_11_08_21:28:59
ferminet_2025_11_08_21:34:33
ferminet_2025_11_08_21:37:38
ferminet_2025_11_08_21:38:07
ferminet_2025_11_08_21:42:42
ferminet_2025_11_08_21:50:53
ferminet_2025_11_08_21:54:29
ferminet_2025_11_08_21:59:52
ferminet_2025_11_08_22:14:58
ferminet_2025_11_08_22:15:51
ferminet_2025_11_08_22:17:07
ferminet_2025_11_08_22:19:05
ferminet_2025_11_08_22:28:58
ferminet_2025_11_09_10:55:40
ferminet_2025_11_09_11:02:05
ferminet_2025_11_09_11:17:54
ferminet_2025_11_09_11:27:38
ferminet_2025_11_09_12:08:52
ferminet_2025_11_09_12:11:49
ferminet_2025_11_09_12:14:57
ferminet_2025_11_09_12:18:58
ferminet_2025_11_09_12:23:07
ferminet_2025_11_09_12:37:09
ferminet_2025_11_09_12:39:48
ferminet_2025_11_09_12:41:03
ferminet_2025_11_09_15:45:04
ferminet_2025_11_09_15:47:47
ferminet_2025_11_09_15:50:27
ferminet_2025_11_09_15:53:17
ferminet_2025_11_09_15:55:43
ferminet_2025_11_09_15:57:46
ferminet_2025_11_09_16:02:20
ferminet_2025_11_09_16:07:21
ferminet_2025_11_09_16:10:53
ferminet_2025_11_09_16:16:52
ferminet_2025_11_09_16:25:16
ferminet_2025_11_09_16:31:21
ferminet_2025_11_09_16:33:37
ferminet_2025_11_09_16:45:00
ferminet_2025_11_09_16:45:30
ferminet_2025_11_09_16:50:44
ferminet_2025_11_09_16:51:34
ferminet_2025_11_09_17:06:44
ferminet_2025_11_09_17:10:51
ferminet_2025_11_09_17:19:09
EOF
)

echo "== Dry-run =="
printf '%s\n' "$FILELIST" | rsync -aHAXnv -s --info=stats1,progress2 --files-from=- ./ "$DST_BASE/"

echo "== Copying =="
printf '%s\n' "$FILELIST" | rsync -aHAX -s --info=stats1,progress2 --files-from=- ./ "$DST_BASE/"

echo "== Verifying by checksum (may list changes on first run) =="
printf '%s\n' "$FILELIST" | rsync -aHAXcni -s --delete --files-from=- ./ "$DST_BASE/"

echo "== Removing originals safely (only files already copied) =="
printf '%s\n' "$FILELIST" | rsync -aHAX -s --remove-source-files --files-from=- ./ "$DST_BASE/"

echo "== Cleaning up now-empty source directories =="
printf '%s\n' "$FILELIST" | while IFS= read -r d; do
  find "$d" -depth -type d -empty -delete || true
done

echo "Done. Moved into: $DST_BASE"