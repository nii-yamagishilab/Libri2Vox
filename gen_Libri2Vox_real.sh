#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash gen_Libri2Vox_real.sh [LIBRITTS_DIR] [VOXCELEB2_DIR]

Arguments (optional, positional):
  LIBRITTS_DIR   Path to LibriTTS root (default: ./data/LibriTTS)
  VOXCELEB2_DIR  Path to VoxCeleb2 root (default: ./data/VoxCeleb2)

You can still use environment variables for advanced options, for example:
  OUTPUT_DIR, METADATA_CSV, SNR_MIN, SNR_MAX, FIXED_LENGTH_SEC, VAL_SPEAKER_COUNT
EOF
  exit 0
fi

if [ "$#" -gt 2 ]; then
  echo "[ERROR] Too many arguments. Run with --help for usage." >&2
  exit 1
fi

LIBRITTS_DIR="${1:-${LIBRITTS_DIR:-$PROJECT_DIR/data/LibriTTS}}"
VOXCELEB2_DIR="${2:-${VOXCELEB2_DIR:-$PROJECT_DIR/data/VoxCeleb2}}"
METADATA_CSV="${METADATA_CSV:-$PROJECT_DIR/assets/metadata/vox2_meta_extended.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output_dataset/real}"

SNR_MIN="${SNR_MIN:--5.0}"
SNR_MAX="${SNR_MAX:-5.0}"
FIXED_LENGTH_SEC="${FIXED_LENGTH_SEC:-6.0}"
VAL_SPEAKER_COUNT="${VAL_SPEAKER_COUNT:-94}"

if [ ! -d "$LIBRITTS_DIR" ]; then
  echo "[ERROR] LIBRITTS_DIR not found: $LIBRITTS_DIR" >&2
  exit 1
fi

if [ ! -d "$VOXCELEB2_DIR" ]; then
  echo "[ERROR] VOXCELEB2_DIR not found: $VOXCELEB2_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

METADATA_ARGS=()
if [ -f "$METADATA_CSV" ]; then
  METADATA_ARGS=(--metadata_csv "$METADATA_CSV")
else
  echo "[WARN] Metadata CSV not found at $METADATA_CSV"
  echo "[WARN] Will fallback to \$VOXCELEB2_DIR/vox2_meta_extended.csv if available."
fi

echo "[INFO] Generating real Libri2Vox dataset..."
echo "[INFO] LIBRITTS_DIR=$LIBRITTS_DIR"
echo "[INFO] VOXCELEB2_DIR=$VOXCELEB2_DIR"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"

"$PYTHON_BIN" "$PROJECT_DIR/gen_dataset.py" \
  --libritts_dir "$LIBRITTS_DIR" \
  --voxceleb2_dir "$VOXCELEB2_DIR" \
  --output_dir "$OUTPUT_DIR" \
  "${METADATA_ARGS[@]}" \
  --snr_min "$SNR_MIN" \
  --snr_max "$SNR_MAX" \
  --fixed_length_sec "$FIXED_LENGTH_SEC" \
  --val_speaker_count "$VAL_SPEAKER_COUNT"

echo "[INFO] Done. Dataset written to: $OUTPUT_DIR"
