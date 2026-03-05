#!/usr/bin/env python3
# Author: Liu Yun
# Copyright: NII Yamagishi lab
# Modified from https://github.com/BakerBunker/SALT by BakerBunker
"""
One-file SALT batch generation script.

Features:
1) Load model with official torch.hub API.
2) Recursively scan input directory for wav files.
3) Save SALT outputs under output directory with same relative paths.
4) Include runtime compatibility logic so this script works even when
   speaker packs are stored as raw Tensor (your local style).
"""

import argparse
import importlib.util
import random
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

DEFAULT_PACK_ZIP_URL = (
    "https://github.com/BakerBunker/SALT/releases/download/1.0.0/librispeech-pack.zip"
)
PY39_UNION_ERROR_KEY = "unsupported operand type(s) for |"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SE_WEIGHTS = str(PROJECT_ROOT / "assets" / "models" / "final_model.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click SALT batch wav generator (single script)."
    )
    parser.add_argument("--input_dir", required=True, help="Input folder containing wav files.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output folder. Default: <input_dir>_salt",
    )
    parser.add_argument(
        "--hub_repo",
        default="BakerBunker/SALT",
        help='Torch hub repo. Default: "BakerBunker/SALT".',
    )
    parser.add_argument(
        "--torch_hub_dir",
        default=None,
        help=(
            "Optional torch hub cache directory. "
            "If not set, torch default cache path is used."
        ),
    )
    parser.add_argument(
        "--speaker_pack_dir",
        default="assets/librispeech-pack",
        help=(
            "Directory containing speaker packs. Relative paths are resolved "
            "from this script directory, and .pack files are scanned recursively."
        ),
    )
    parser.add_argument(
        "--pack_glob",
        default="*.pack",
        help="Glob pattern used to match speaker packs (recursive scan by default).",
    )
    parser.add_argument(
        "--auto_download_packs",
        dest="auto_download_packs",
        action="store_true",
        default=True,
        help="Auto-download official librispeech speaker packs if not found.",
    )
    parser.add_argument(
        "--no_auto_download_packs",
        dest="auto_download_packs",
        action="store_false",
        help="Disable auto download of speaker packs.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate. Input wavs with other sample rates are auto-resampled.",
    )
    parser.add_argument(
        "--clip_seconds",
        type=float,
        default=0.0,
        help="If > 0, crop long audio to this many seconds. No zero-padding is applied.",
    )
    parser.add_argument(
        "--random_start",
        action="store_true",
        help="For long wav, randomly choose crop start. Default is from beginning.",
    )
    parser.add_argument(
        "--mix_speakers",
        type=int,
        default=4,
        help="How many random speakers to mix for each input file.",
    )
    parser.add_argument(
        "--self_only",
        action="store_true",
        help="Use only original speaker features (self=1.0) then vocode with HiFiGAN.",
    )
    parser.add_argument(
        "--preservation_factor",
        type=float,
        default=0.8,
        help="Weight kept from original speaker identity (0~1, must be <1).",
    )
    parser.add_argument("--topk", type=int, default=4, help="topk for SALT matching.")
    parser.add_argument(
        "--tgt_loudness_db",
        type=float,
        default=-27.0,
        help="Target output loudness in dB.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="Skip failed files and continue.",
    )
    parser.add_argument(
        "--enable_se_remix",
        dest="enable_se_remix",
        action="store_true",
        default=True,
        help=(
            "Enable pipeline: noisy -> SE separate -> SALT(clean_est) -> remix with estimated noise "
            "(enabled by default)."
        ),
    )
    parser.add_argument(
        "--disable_se_remix",
        dest="enable_se_remix",
        action="store_false",
        help="Disable SE-remix and run SALT directly on input audio.",
    )
    parser.add_argument(
        "--se_weights",
        default=DEFAULT_SE_WEIGHTS,
        help="SE model weights path used when --enable_se_remix is set.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.mix_speakers <= 0:
        raise ValueError("--mix_speakers must be > 0.")
    if not (0 <= args.preservation_factor < 1):
        raise ValueError("--preservation_factor must satisfy 0 <= value < 1.")


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device_arg


def resolve_pack_dir(path_text: str) -> Path:
    pack_dir = Path(path_text).expanduser()
    if pack_dir.is_absolute():
        return pack_dir.resolve()
    script_dir = Path(__file__).resolve().parent
    return (script_dir / pack_dir).resolve()


def configure_torch_hub_dir(torch_hub_dir: Optional[str]) -> Path:
    if torch_hub_dir:
        custom_dir = Path(torch_hub_dir).expanduser().resolve()
        custom_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(custom_dir))
        print(f"Using custom torch hub dir: {custom_dir}")

    cache_root = Path(torch.hub.get_dir()).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def normalize_clip_seconds(clip_seconds: float) -> Optional[float]:
    if clip_seconds <= 0:
        return None
    return clip_seconds


def collect_wavs(input_dir: Path) -> List[Path]:
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"
    )


def load_audio(
    file_path: Path,
    sample_rate: int,
    clip_seconds: Optional[float],
    start_from_beginning: bool,
) -> Tuple[torch.FloatTensor, bool]:
    audio, src_sr = sf.read(str(file_path), dtype="float32")
    was_resampled = False

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if src_sr != sample_rate:
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        wav = torchaudio.functional.resample(
            wav, orig_freq=src_sr, new_freq=sample_rate
        )
        audio = wav.squeeze(0).cpu().numpy()
        was_resampled = True

    if clip_seconds is not None:
        target_samples = int(round(clip_seconds * sample_rate))
        if audio.shape[0] > target_samples:
            if start_from_beginning:
                start_idx = 0
            else:
                start_idx = random.randint(0, audio.shape[0] - target_samples)
            audio = audio[start_idx : start_idx + target_samples]

    return torch.from_numpy(audio.astype(np.float32)), was_resampled


def find_pack_files(pack_dir: Path, pack_glob: str) -> List[Path]:
    if not pack_dir.exists():
        return []
    if any(token in pack_glob for token in ("**", "/", "\\")):
        matched = pack_dir.glob(pack_glob)
    else:
        matched = pack_dir.rglob(pack_glob)
    return sorted(p.resolve() for p in matched if p.is_file())


def download_default_packs(pack_dir: Path) -> Path:
    pack_dir.mkdir(parents=True, exist_ok=True)
    print(f"No speaker pack found, downloading: {DEFAULT_PACK_ZIP_URL}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "librispeech-pack.zip"
        urllib.request.urlretrieve(DEFAULT_PACK_ZIP_URL, str(zip_path))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(pack_dir.parent))

    if any(pack_dir.glob("*.pack")):
        return pack_dir

    # Fallback: auto-detect extracted pack directory.
    candidate_dirs = sorted(
        {p.parent.resolve() for p in pack_dir.parent.rglob("*.pack")},
        key=str,
    )
    if candidate_dirs:
        detected_dir = candidate_dirs[0]
        print(f"Auto-detected speaker pack directory: {detected_dir}")
        return detected_dir
    return pack_dir


def patch_torch_hub_cache_for_py39(hub_repo: str, cache_root: Path) -> bool:
    """
    For Python < 3.10, patch cached torch hub matcher.py by adding
    `from __future__ import annotations`, so `float | None` hints do not
    evaluate at import time.
    """
    if sys.version_info >= (3, 10):
        return False
    if "/" not in hub_repo:
        return False
    if Path(hub_repo).expanduser().exists():
        # Do not mutate user-provided local repositories.
        return False

    repo_key = hub_repo.replace("/", "_")
    repo_dirs = sorted(p for p in cache_root.glob(f"{repo_key}_*") if p.is_dir())
    if not repo_dirs:
        return False

    patched_any = False
    for repo_dir in repo_dirs:
        matcher_file = repo_dir / "matcher.py"
        if not matcher_file.exists():
            continue
        content = matcher_file.read_text(encoding="utf-8")
        future_import = "from __future__ import annotations"
        if future_import in content:
            continue
        matcher_file.write_text(f"{future_import}\n{content}", encoding="utf-8")
        patched_any = True
    return patched_any


def _iter_speaker_weights(speaker_dict) -> Iterable[Tuple[str, float]]:
    if isinstance(speaker_dict, dict):
        for name, weight in speaker_dict.items():
            yield str(name), float(weight)
        return

    if hasattr(speaker_dict, "itertuples"):
        for row in speaker_dict.itertuples(index=False):
            yield str(row.speaker), float(row.weight)
        return

    raise TypeError(
        "speaker_dict must be either dict or pandas.DataFrame-like object "
        "with columns 'speaker' and 'weight'."
    )


def interpolate_compat(
    anon,
    wav: torch.Tensor,
    speaker_dict,
    topk: int,
    tgt_loudness_db: float,
) -> torch.Tensor:
    """
    Runtime-compatible interpolation:
    - Works with official SALT hub code.
    - Works with your modified .pack format (Tensor or tuple).
    - Supports explicit tgt_loudness_db without editing source files.
    """
    src_feat = anon.knnvc.get_features(wav)
    tgt_feat = torch.zeros_like(src_feat)

    for speaker_name, weight in _iter_speaker_weights(speaker_dict):
        if speaker_name == "self":
            tgt_feat += src_feat * weight
            continue

        if speaker_name not in anon.pool:
            raise KeyError(f"Speaker '{speaker_name}' not found in loaded pool.")

        speaker_entry = anon.pool[speaker_name]
        matching_set = speaker_entry[0] if isinstance(speaker_entry, tuple) else speaker_entry
        matched_feat = anon.knnvc.match_feat(src_feat, matching_set, topk=topk).squeeze(0)
        tgt_feat += matched_feat * weight

    return anon.knnvc.feat_to_wav(tgt_feat.unsqueeze(0), tgt_loudness_db=tgt_loudness_db)


def load_salt_model(args: argparse.Namespace):
    device = resolve_device(args.device)
    cache_root = configure_torch_hub_dir(args.torch_hub_dir)
    print(f"Loading SALT from torch hub: {args.hub_repo} (device={device})")
    prepatched = patch_torch_hub_cache_for_py39(args.hub_repo, cache_root)
    if prepatched:
        print("[INFO] Applied Python 3.9 compatibility patch to cached SALT code.")

    try:
        anon = torch.hub.load(
            args.hub_repo,
            "salt",
            trust_repo=True,
            pretrained=True,
            base=True,
            device=device,
        )
    except TypeError as exc:
        if PY39_UNION_ERROR_KEY not in str(exc):
            raise
        repatched = patch_torch_hub_cache_for_py39(args.hub_repo, cache_root)
        if not repatched:
            raise RuntimeError(
                "Detected Python<3.10 annotation issue while loading SALT from torch hub. "
                "Please rerun once, or use Python>=3.10."
            ) from exc
        print("[INFO] Patched cached SALT matcher.py for Python 3.9, retrying load...")
        anon = torch.hub.load(
            args.hub_repo,
            "salt",
            trust_repo=True,
            pretrained=True,
            base=True,
            device=device,
        )

    if args.self_only:
        print("[INFO] Self-only mode enabled. Skip loading speaker packs.")
        return anon

    pack_dir = resolve_pack_dir(args.speaker_pack_dir)
    pack_files = find_pack_files(pack_dir, args.pack_glob)
    if not pack_files and args.auto_download_packs:
        try:
            pack_dir = download_default_packs(pack_dir)
            pack_files = find_pack_files(pack_dir, args.pack_glob)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Auto-download speaker packs failed: {exc}")

    if not pack_files:
        raise FileNotFoundError(
            "No speaker pack files found.\n"
            f"  directory: {pack_dir}\n"
            f"  glob: {args.pack_glob}\n"
            "Use --speaker_pack_dir to point to your pack folder, or enable --auto_download_packs."
        )

    for pack_file in tqdm(pack_files, desc="Loading speaker packs", unit="pack"):
        anon.add_speaker(name=pack_file.stem, preprocessed_file=str(pack_file))

    print(f"Loaded {len(pack_files)} speaker packs from: {pack_dir}")
    return anon


def load_se_module():
    candidate_paths = [
        PROJECT_ROOT / "separatenoise_and_remix.py",
        PROJECT_ROOT.parent / "separatenoise_and_remix.py",
        PROJECT_ROOT.parent / "eval" / "separatenoise_and_remix.py",
    ]
    module_path = next((path for path in candidate_paths if path.exists()), None)
    if module_path is None:
        searched = "\n".join(f"  - {path}" for path in candidate_paths)
        raise FileNotFoundError(
            "SE module not found. Checked:\n"
            f"{searched}"
        )

    spec = importlib.util.spec_from_file_location("separatenoise_and_remix_module", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import SE module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_se_engine(args: argparse.Namespace):
    if not args.enable_se_remix:
        return None, None

    se_weights = Path(args.se_weights).expanduser().resolve()
    if not se_weights.is_file():
        raise FileNotFoundError(
            f"SE weights not found: {se_weights}\n"
            "Please pass a valid --se_weights path when using --enable_se_remix."
        )

    se_module = load_se_module()
    print(f"[INFO] SE remix mode enabled. Weights: {se_weights}")
    se_engine = se_module.NoiseSeparatorRemixer(
        model_weights=se_weights,
        sample_rate=args.sample_rate,
        device=args.device,
    )
    return se_module, se_engine


def _compute_amplitude(waveform: np.ndarray) -> float:
    return float(np.mean(np.abs(waveform)))


def _align_to_length(waveform: np.ndarray, target_length: int) -> np.ndarray:
    if len(waveform) == target_length:
        return waveform
    if len(waveform) > target_length:
        return waveform[:target_length]
    return np.pad(waveform, (0, target_length - len(waveform)), mode="constant")


def _separate_then_salt_and_remix_local(
    engine,
    noisy_audio: np.ndarray,
    salt_speech_fn,
    clip_audio: bool,
):
    noise_est, clean_est = engine.separate(noisy_audio.astype(np.float32))
    salted_clean = salt_speech_fn(clean_est.astype(np.float32))
    salted_clean = _align_to_length(salted_clean, len(noise_est))
    noisy_aligned = _align_to_length(noisy_audio.astype(np.float32), len(noise_est))
    ori_amp = _compute_amplitude(noisy_aligned)
    speech_amp = _compute_amplitude(salted_clean)
    scale = ori_amp / max(speech_amp, 1e-8)
    remixed = salted_clean * scale + noise_est
    if clip_audio:
        remixed = np.clip(remixed, -1.0, 1.0)
    return (
        remixed.astype(np.float32),
        noise_est.astype(np.float32),
        salted_clean.astype(np.float32),
        float(scale),
    )


def process_and_save_files(args: argparse.Namespace, anon) -> None:
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_dir.with_name(f"{input_dir.name}_salt")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    file_paths = collect_wavs(input_dir)
    if not file_paths:
        print(f"No wav files found in: {input_dir}")
        return

    clip_seconds = normalize_clip_seconds(args.clip_seconds)
    start_from_beginning = not args.random_start
    se_module, se_engine = build_se_engine(args)

    speakers_per_mix = 0
    if not args.self_only:
        pool_size = len(getattr(anon, "pool", {}))
        if pool_size <= 0:
            raise RuntimeError("No speakers loaded into anonymizer pool.")
        speakers_per_mix = min(args.mix_speakers, pool_size)
        if speakers_per_mix < args.mix_speakers:
            print(
                f"[WARN] Requested mix_speakers={args.mix_speakers}, but only {pool_size} "
                f"speakers loaded. Using {speakers_per_mix}."
            )

    print(f"Found {len(file_paths)} wav files.")
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")

    errors = 0
    resampled_files = 0
    for file_path in tqdm(file_paths, desc="Generating SALT wavs", unit="file"):
        try:
            audio, was_resampled = load_audio(
                file_path=file_path,
                sample_rate=args.sample_rate,
                clip_seconds=clip_seconds,
                start_from_beginning=start_from_beginning,
            )
            if was_resampled:
                resampled_files += 1
            if args.self_only:
                speaker = {"self": 1.0}
            else:
                speaker = anon.get_random_speaker(
                    speakers=speakers_per_mix,
                    preservation_factor=args.preservation_factor,
                )
            if args.enable_se_remix:
                noisy_np = audio.detach().cpu().numpy().astype(np.float32)

                def salt_speech_fn(clean_est: np.ndarray) -> np.ndarray:
                    clean_tensor = torch.from_numpy(clean_est.astype(np.float32))
                    salted_tensor = interpolate_compat(
                        anon=anon,
                        wav=clean_tensor,
                        speaker_dict=speaker,
                        topk=args.topk,
                        tgt_loudness_db=args.tgt_loudness_db,
                    )
                    return salted_tensor.detach().cpu().numpy().astype(np.float32)

                if hasattr(se_module, "separate_then_salt_and_remix"):
                    remixed, _, _, _ = se_module.separate_then_salt_and_remix(
                        engine=se_engine,
                        noisy_audio=noisy_np,
                        salt_speech_fn=salt_speech_fn,
                        clip_audio=True,
                    )
                else:
                    remixed, _, _, _ = _separate_then_salt_and_remix_local(
                        engine=se_engine,
                        noisy_audio=noisy_np,
                        salt_speech_fn=salt_speech_fn,
                        clip_audio=True,
                    )
                wav = torch.from_numpy(remixed.astype(np.float32))
            else:
                wav = interpolate_compat(
                    anon=anon,
                    wav=audio,
                    speaker_dict=speaker,
                    topk=args.topk,
                    tgt_loudness_db=args.tgt_loudness_db,
                )

            input_samples = int(audio.shape[0])
            if int(wav.shape[0]) != input_samples:
                # Match output length to input length without zero-padding.
                wav = F.interpolate(
                    wav.view(1, 1, -1),
                    size=input_samples,
                    mode="linear",
                    align_corners=False,
                ).view(-1)

            out_path = output_dir / file_path.relative_to(input_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), wav.detach().cpu().numpy(), args.sample_rate)
        except Exception as exc:  # pylint: disable=broad-except
            errors += 1
            if args.skip_errors:
                print(f"[WARN] Skip {file_path}: {exc}")
                continue
            raise

    if resampled_files:
        print(f"Auto-resampled {resampled_files} file(s) to {args.sample_rate} Hz.")

    if errors:
        print(f"Done with {errors} skipped files.")
    else:
        print("Done without errors.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    anon = load_salt_model(args)
    process_and_save_files(args, anon)


if __name__ == "__main__":
    main()
