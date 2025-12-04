# Create the script
cat > safe_move_minimalChern.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

# You can override on run:  DST_BASE=/somewhere ./safe_move_minimalChern.sh
DST_BASE="${DST_BASE:-/ceph/submit/data/user/a/ahmed95/minimalChern_NN/}"

SRC_BASE="/work/submit/ahmed95"
mkdir -p "$DST_BASE"

# List of directories to move (kept verbatim; colons are fine here)
LIST=$(cat <<'EOF'
minimalChern_NN/ferminet_2025_10_06_18:32:56/
minimalChern_NN/ferminet_2025_10_06_20:31:05/
minimalChern_NN/ferminet_2025_10_06_22:53:29/
minimalChern_NN/ferminet_2025_10_07_11:14:53/
minimalChern_NN/ferminet_2025_10_09_01:18:05/
minimalChern_NN/ferminet_2025_10_09_01:18:29/
minimalChern_NN/ferminet_2025_10_10_22:09:39/
minimalChern_NN/ferminet_2025_10_10_22:17:43/
minimalChern_NN/ferminet_2025_10_12_11:32:12/
minimalChern_NN/ferminet_2025_10_12_15:12:48/
minimalChern_NN/ferminet_2025_10_12_19:05:28/
minimalChern_NN/ferminet_2025_10_12_20:43:58/
minimalChern_NN/ferminet_2025_10_14_15:50:28/
minimalChern_NN/ferminet_2025_10_14_17:15:30/
minimalChern_NN/ferminet_2025_10_15_13:36:13/
minimalChern_NN/ferminet_2025_10_15_14:58:47/
minimalChern_NN/ferminet_2025_10_15_17:07:52/
minimalChern_NN/ferminet_2025_10_16_21:35:30/
minimalChern_NN/ferminet_2025_10_17_06:38:30/
minimalChern_NN/ferminet_2025_10_17_07:25:00/
minimalChern_NN/ferminet_2025_10_17_11:33:42/
minimalChern_NN/ferminet_2025_10_17_15:57:28/
minimalChern_NN/ferminet_2025_10_17_16:02:58/
minimalChern_NN/ferminet_2025_10_17_21:29:54/
minimalChern_NN/ferminet_2025_10_17_23:15:43/
minimalChern_NN/ferminet_2025_10_18_09:12:02/
minimalChern_NN/ferminet_2025_10_18_09:22:57/
minimalChern_NN/ferminet_2025_10_18_10:04:11/
EOF
)

cd "$SRC_BASE"

echo "== Dry-run =="
rsync -aHAXnv -s --info=stats1,progress2 --files-from=<(printf "%s\n" "$LIST") ./ "$DST_BASE/"

echo "== Copying =="
rsync -aHAX -s --info=stats1,progress2 --files-from=<(printf "%s\n" "$LIST") ./ "$DST_BASE/"

echo "== Verifying by checksum (should print nothing) =="
rsync -aHAXcni -s --delete --files-from=<(printf "%s\n" "$LIST") ./ "$DST_BASE/"

echo "== Removing originals safely (only files already copied) =="
rsync -aHAX -s --remove-source-files --files-from=<(printf "%s\n" "$LIST") ./ "$DST_BASE/"

echo "== Cleaning up now-empty source directories =="
printf "%s\n" "$LIST" | while IFS= read -r d; do
  find "$d" -depth -type d -empty -delete || true
done

echo "Done. Moved into: $DST_BASE/minimalChern_NN/"
SH

# Make it executable and run
chmod +x safe_move_minimalChern.sh
./safe_move_minimalChern.sh
# Or choose a different destination:
# DST_BASE="/scratch/ahmed95/ARCHIVE_minimalChern_runs" ./safe_move_minimalChern.sh