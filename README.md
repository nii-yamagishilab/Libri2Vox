# Libri2Vox

This repository provides the scripts of Libri2Vox, which is via generation of speech-mixture triplets from `LibriTTS` (target speech) and `VoxCeleb2` (interference speech).

Each triplet contains:

- `target.wav`
- `reference.wav`
- `mix.wav`

This repository provides the material for:
```
"Target Speaker Extraction with Diverse Speaker Conditions and Synthetic Data" 
Yun Liu, Xuechen Liu, Xiaoxiao Miao, and Junichi Yamagishi
in APSIPA Jour. On Signal Processing, vol. 14, 2025.
DOI: 10.1561/116.20250054
```

[Paper link, Arxiv](https://arxiv.org/abs/2412.12512)

## Disclaimer

This repository does **NOT** redistribute original LibriTTS/VoxCeleb2 audio. Users must prepare both datasets locally and follow the original dataset licenses and terms of use.

## Data Sources

- LibriTTS: <https://www.openslr.org/60/>
- VoxCeleb2: <https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html>
- SALT (for `syn_salt`): <https://github.com/BakerBunker/SALT>

Expected roots:

- `LIBRITTS_DIR` with `train-clean-100`, `train-clean-360`, `dev-clean`, `test-clean`
- `VOXCELEB2_DIR` with `dev`, `test`

Metadata (recommended):

- default: `assets/metadata/vox2_meta_extended.csv`
- fallback: `${VOXCELEB2_DIR}/vox2_meta_extended.csv`

## Quick Start

```bash
conda env create -f environment.yml
conda activate libri2vox
```

```bash
LIBRITTS_DIR="/your/path/to/LibriTTS"
VOXCELEB2_DIR="/your/path/to/VoxCeleb2"
```

Generate real-interference dataset:

```bash
bash gen_Libri2Vox_real.sh "$LIBRITTS_DIR" "$VOXCELEB2_DIR"
```

Generate SALT-based synthetic-interference dataset:

```bash
bash gen_Libri2Vox_syn_salt.sh "$LIBRITTS_DIR" "$VOXCELEB2_DIR"
```

Script help:

```bash
bash gen_Libri2Vox_real.sh --help
bash gen_Libri2Vox_syn_salt.sh --help
```

## Optional Settings

Both scripts support environment-variable overrides.

Common options:

- `OUTPUT_DIR`
- `METADATA_CSV`
- `SNR_MIN`, `SNR_MAX`
- `FIXED_LENGTH_SEC`
- `DEVICE` (`auto/cuda/cpu`)
- `ENABLE_SE_REMIX` (`1/0`, default: `1` in `gen_Libri2Vox_syn_salt.sh`)
- `SE_WEIGHTS` (optional custom path for SE checkpoint)

`DEVICE=auto` uses GPU automatically when CUDA is available.

SE remix is enabled by default in the `syn_salt` pipeline.

Quick check (look for SE-remix logs):

```bash
ENABLE_SE_REMIX=1 \
CLIP_SECONDS=2 MIX_SPEAKERS=1 \
bash gen_Libri2Vox_syn_salt.sh "$LIBRITTS_DIR" "$VOXCELEB2_DIR"
```

You should see logs such as:

- `[INFO] SE remix enabled by default (ENABLE_SE_REMIX=1)`
- `[INFO] SE remix mode enabled. Weights: ...`

Disable SE remix if needed:

```bash
ENABLE_SE_REMIX=0 bash gen_Libri2Vox_syn_salt.sh "$LIBRITTS_DIR" "$VOXCELEB2_DIR"
```

## Optional Preprocessing (`sv56`)

`sv56` was used in the paper as an optional level-related preprocessing step.  
In our tests, the impact was small. If needed, preprocess both datasets before running this pipeline.

- sv56 tool: <https://github.com/openitu/STL/tree/dev/src/sv56>

## Output

Default outputs:

- `output_dataset/real`
- `output_dataset/syn_salt`

Each split (`train/val/test`) contains multiple `triplet*` folders with:

- `target.wav`
- `reference.wav`
- `mix.wav`

## License

This project is released under the MIT License.
See `LICENSE` for the full license text.

The repository only contains scripts/metadata and does not redistribute LibriTTS or VoxCeleb2.
Please also follow the original dataset licenses and terms of use.

## Acknowledgement

This study is partially supported by MEXT KAKENHI Grants (24K21324)
and JST, the establishment of university fellowships towards the creation of science
technology innovation (JPMJFS2136)

## Citation

If you find this project useful, please cite:

```bibtex
@article{Liu2024Libri2Vox,
  author    = {Yun Liu and Xuechen Liu and Xiaoxiao Miao and Junichi Yamagishi},
  title     = {Target Speaker Extractor Training with Diverse Speaker Conditions and Synthetic Data},
  journal   = {APSIPA Transactions on Signal and Information Processing},
  volume    = {14},
  number    = {1},
  pages     = {e30},
  year      = {2025},
  doi       = {10.1561/116.20250054}
}
```

---
Written by [Yun Liu](https://github.com/MaNatsu8023)
