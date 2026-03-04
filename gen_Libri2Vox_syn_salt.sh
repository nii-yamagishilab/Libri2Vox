#!/usr/bin/env bash
# Author: Liu Yun
# Copyright: NII Yamagishi lab
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash gen_Libri2Vox_syn_salt.sh [LIBRITTS_DIR] [VOXCELEB2_DIR]

Arguments (optional, positional):
  LIBRITTS_DIR   Path to LibriTTS root (default: ./data/LibriTTS)
  VOXCELEB2_DIR  Path to VoxCeleb2 root (default: ./data/VoxCeleb2)

You can still use environment variables for advanced options, for example:
  SALT_VOXCELEB2_DIR, OUTPUT_DIR, METADATA_CSV, DEVICE, MIX_SPEAKERS,
  PRESERVATION_FACTOR, TOPK, TGT_LOUDNESS_DB, CLIP_SECONDS,
  ENABLE_SE_REMIX (default: 1), SE_WEIGHTS
EOF
  exit 0
fi

if [ "$#" -gt 2 ]; then
  echo "[ERROR] Too many arguments. Run with --help for usage." >&2
  exit 1
fi

LIBRITTS_DIR="${1:-${LIBRITTS_DIR:-$PROJECT_DIR/data/LibriTTS}}"
VOXCELEB2_DIR="${2:-${VOXCELEB2_DIR:-$PROJECT_DIR/data/VoxCeleb2}}"
SALT_VOXCELEB2_DIR="${SALT_VOXCELEB2_DIR:-$PROJECT_DIR/data/VoxCeleb2_salt}"
METADATA_CSV="${METADATA_CSV:-$PROJECT_DIR/assets/metadata/vox2_meta_extended.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output_dataset/syn_salt}"

SNR_MIN="${SNR_MIN:--5.0}"
SNR_MAX="${SNR_MAX:-5.0}"
FIXED_LENGTH_SEC="${FIXED_LENGTH_SEC:-6.0}"
VAL_SPEAKER_COUNT="${VAL_SPEAKER_COUNT:-94}"

DEVICE="${DEVICE:-auto}"
MIX_SPEAKERS="${MIX_SPEAKERS:-4}"
PRESERVATION_FACTOR="${PRESERVATION_FACTOR:-0.8}"
TOPK="${TOPK:-4}"
TGT_LOUDNESS_DB="${TGT_LOUDNESS_DB:--27.0}"
CLIP_SECONDS="${CLIP_SECONDS:-0.0}"
ENABLE_SE_REMIX="${ENABLE_SE_REMIX:-1}"
SE_WEIGHTS="${SE_WEIGHTS:-}"

SE_REMIX_FLAG=(--enable_se_remix)
if [ "$ENABLE_SE_REMIX" = "0" ] || [ "$ENABLE_SE_REMIX" = "false" ] || [ "$ENABLE_SE_REMIX" = "FALSE" ]; then
  SE_REMIX_FLAG=(--disable_se_remix)
fi

SE_WEIGHTS_ARGS=()
if [ -n "$SE_WEIGHTS" ]; then
  SE_WEIGHTS_ARGS=(--se_weights "$SE_WEIGHTS")
fi

if [ ! -d "$LIBRITTS_DIR" ]; then
  echo "[ERROR] LIBRITTS_DIR not found: $LIBRITTS_DIR" >&2
  exit 1
fi

if [ ! -d "$VOXCELEB2_DIR" ]; then
  echo "[ERROR] VOXCELEB2_DIR not found: $VOXCELEB2_DIR" >&2
  exit 1
fi

if [ "$VOXCELEB2_DIR" = "$SALT_VOXCELEB2_DIR" ]; then
  echo "[ERROR] SALT_VOXCELEB2_DIR must be different from VOXCELEB2_DIR." >&2
  exit 1
fi

mkdir -p "$SALT_VOXCELEB2_DIR"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Step 1/2: Converting VoxCeleb2 wav to SALT wav..."
echo "[INFO] INPUT VOXCELEB2_DIR=$VOXCELEB2_DIR"
echo "[INFO] OUTPUT SALT_VOXCELEB2_DIR=$SALT_VOXCELEB2_DIR"
echo "[INFO] SE remix enabled by default (ENABLE_SE_REMIX=$ENABLE_SE_REMIX)"

"$PYTHON_BIN" "$PROJECT_DIR/gen_salt_audio.py" \
  --input_dir "$VOXCELEB2_DIR" \
  --output_dir "$SALT_VOXCELEB2_DIR" \
  --device "$DEVICE" \
  --mix_speakers "$MIX_SPEAKERS" \
  --preservation_factor "$PRESERVATION_FACTOR" \
  --topk "$TOPK" \
  --tgt_loudness_db "$TGT_LOUDNESS_DB" \
  --clip_seconds "$CLIP_SECONDS" \
  "${SE_REMIX_FLAG[@]}" \
  "${SE_WEIGHTS_ARGS[@]}" \
  --skip_errors

echo "[INFO] Step 2/2: Generating syn_salt Libri2Vox dataset..."

METADATA_ARGS=()
if [ -f "$METADATA_CSV" ]; then
  METADATA_ARGS=(--metadata_csv "$METADATA_CSV")
else
  echo "[WARN] Metadata CSV not found at $METADATA_CSV"
  echo "[WARN] Will fallback to \$SALT_VOXCELEB2_DIR/vox2_meta_extended.csv if available."
fi

"$PYTHON_BIN" "$PROJECT_DIR/gen_dataset.py" \
  --libritts_dir "$LIBRITTS_DIR" \
  --voxceleb2_dir "$SALT_VOXCELEB2_DIR" \
  --output_dir "$OUTPUT_DIR" \
  "${METADATA_ARGS[@]}" \
  --snr_min "$SNR_MIN" \
  --snr_max "$SNR_MAX" \
  --fixed_length_sec "$FIXED_LENGTH_SEC" \
  --val_speaker_count "$VAL_SPEAKER_COUNT"

echo "[INFO] Done. Dataset written to: $OUTPUT_DIR"
echo "[INFO] Original VOXCELEB2_DIR is untouched: $VOXCELEB2_DIR"
