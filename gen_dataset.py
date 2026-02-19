import argparse
import os
import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA_CSV = PROJECT_ROOT / "assets" / "metadata" / "vox2_meta_extended.csv"


def list_wav_files(base_dir):
    wav_files = []
    if not os.path.isdir(base_dir):
        return wav_files
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files


def speaker_id_from_path(audio_path):
    return audio_path.split("/")[-3]


def get_speaker_files(all_files):
    speakers = {}
    for file in all_files:
        speaker_id = speaker_id_from_path(file)
        speakers[speaker_id] = speakers.get(speaker_id, []) + [file]
    return speakers


def load_audio(audio_path, sr=16000):
    audio, _ = librosa.load(audio_path, sr=sr)
    duration = len(audio) / sr
    return audio, duration


def adjust_audio_length(audio, length_samples):
    if len(audio) > length_samples:
        start_idx = np.random.randint(0, len(audio) - length_samples)
        audio = audio[start_idx : start_idx + length_samples]
    elif len(audio) < length_samples:
        padding = np.zeros(length_samples - len(audio), dtype=audio.dtype)
        audio = np.concatenate((audio, padding))
    return audio


def get_audio_duration(audio_path):
    try:
        info = sf.info(audio_path)
        if info.samplerate > 0:
            return info.frames / float(info.samplerate)
    except Exception:
        pass
    _, duration = load_audio(audio_path, sr=16000)
    return duration


def concatenate_audios(audio_paths, min_length, sr=16000):
    total_audio = []
    total_duration = 0.0
    for path in audio_paths:
        audio, duration = load_audio(path, sr=sr)
        total_audio.append(audio)
        total_duration += duration
        if total_duration >= min_length:
            break
    target_len = int(min_length * sr)
    if not total_audio:
        return np.zeros(target_len, dtype=np.float32)
    concatenated_audio = np.concatenate(total_audio)
    if len(concatenated_audio) < target_len:
        padding = np.zeros(target_len - len(concatenated_audio), dtype=concatenated_audio.dtype)
        concatenated_audio = np.concatenate((concatenated_audio, padding))
    return concatenated_audio[:target_len]


def choose_reference_audio(reference_candidates, sr=16000, min_length=6, max_length=15):
    co = 0
    while True:
        reference_path = random.choice(reference_candidates)
        reference_audio, reference_duration = load_audio(reference_path, sr=sr)
        co += 1
        if reference_duration >= min_length:
            if reference_duration > max_length:
                reference_audio = reference_audio[: int(max_length * sr)]
            return reference_audio
        if co > 100:
            return concatenate_audios(reference_candidates, min_length, sr=sr)


def process_reference_train(reference_candidates, sr=16000, min_duration=10, max_duration=15):
    total_duration = 0.0
    refs_signal = None
    used_files = set()
    while total_duration < min_duration:
        available_files = [f for f in reference_candidates if f not in used_files]
        if not available_files:
            available_files = reference_candidates
        file_path = random.choice(available_files)
        signal, _ = librosa.load(file_path, sr=sr)
        if refs_signal is None:
            refs_signal = signal
        else:
            refs_signal = np.append(refs_signal, signal)
        total_duration = len(refs_signal) / sr
        used_files.add(file_path)
    if total_duration > max_duration:
        refs_signal = refs_signal[: int(max_duration * sr)]
    return refs_signal


def align_interference_to_target(target_audio, interference_audio):
    target_len = len(target_audio)
    if len(interference_audio) > target_len:
        interference_audio = interference_audio[:target_len]
    elif len(interference_audio) < target_len:
        padding = np.zeros(target_len - len(interference_audio), dtype=interference_audio.dtype)
        interference_audio = np.concatenate((interference_audio, padding))
    return interference_audio


def adjust_interference_for_snr(target_audio, interference_audio, desired_snr):
    signal_power = np.sum(target_audio ** 2.0)
    noise_power = np.sum(interference_audio ** 2.0)
    if noise_power <= 0:
        return interference_audio
    alpha = np.sqrt(signal_power / (noise_power * (10.0 ** (desired_snr / 10.0))))
    return interference_audio * alpha


def resolve_metadata_csv(metadata_csv, voxceleb2_dir):
    candidates = []
    if metadata_csv:
        user_path = Path(metadata_csv).expanduser()
        candidates.append(user_path)
        if not user_path.is_absolute():
            candidates.append(PROJECT_ROOT / user_path)
    candidates.append(os.path.join(voxceleb2_dir, "vox2_meta_extended.csv"))
    candidates.append(DEFAULT_METADATA_CSV)
    for path in candidates:
        if path and os.path.isfile(str(path)):
            return str(path)
    raise FileNotFoundError("metadata csv not found")


def load_gender_speakers(metadata_csv, metadata_set):
    metadata = pd.read_csv(metadata_csv, delimiter="\t")
    metadata["Gender"] = metadata["Gender"].astype(str).str.strip()
    metadata["VoxCeleb2 ID"] = metadata["VoxCeleb2 ID"].astype(str).str.strip()
    metadata["Set"] = metadata["Set"].astype(str).str.strip()
    male_speakers = metadata[
        (metadata["Gender"] == "m") & (metadata["Set"] == metadata_set)
    ]["VoxCeleb2 ID"].tolist()
    female_speakers = metadata[
        (metadata["Gender"] == "f") & (metadata["Set"] == metadata_set)
    ]["VoxCeleb2 ID"].tolist()
    return male_speakers, female_speakers


def filter_train_targets(target_files):
    by_speaker_all = get_speaker_files(target_files)
    filtered_targets = [
        path
        for path in target_files
        if len(by_speaker_all.get(speaker_id_from_path(path), [])) >= 4
    ]
    return filtered_targets, by_speaker_all


def filter_eval_targets(target_files):
    by_speaker_all = get_speaker_files(target_files)
    filtered_targets = []
    for path in target_files:
        duration = get_audio_duration(path)
        if duration >= 2.0:
            filtered_targets.append(path)
    return filtered_targets, by_speaker_all


def choose_interference_speaker(male_speakers, female_speakers, speaker_usage):
    if male_speakers and female_speakers:
        if speaker_usage["male"] <= speaker_usage["female"]:
            speaker_usage["male"] += 1
            return random.choice(male_speakers)
        speaker_usage["female"] += 1
        return random.choice(female_speakers)
    if male_speakers:
        speaker_usage["male"] += 1
        return random.choice(male_speakers)
    if female_speakers:
        speaker_usage["female"] += 1
        return random.choice(female_speakers)
    raise ValueError("no available male/female speakers for interference")


def generate_split_triplets(
    target_files,
    target_by_speaker_all,
    interference_speaker_files,
    male_speakers,
    female_speakers,
    output_split_dir,
    snr_min,
    snr_max,
    split_name,
    fixed_length_sec=6.0,
    sr=16000,
):
    os.makedirs(output_split_dir, exist_ok=True)
    speaker_usage = {"male": 0, "female": 0}
    triplet_idx = 1
    for target_path in target_files:
        target_audio, _ = load_audio(target_path, sr=sr)
        speaker_id = speaker_id_from_path(target_path)
        same_speaker_files = target_by_speaker_all.get(speaker_id, [])
        reference_candidates = [path for path in same_speaker_files if path != target_path]
        if not reference_candidates:
            continue
        if split_name == "train":
            reference_audio = process_reference_train(
                reference_candidates, sr=sr, min_duration=10, max_duration=15
            )
        else:
            reference_audio = choose_reference_audio(
                reference_candidates, sr=sr, min_length=6, max_length=15
            )
        interference_speaker_id = choose_interference_speaker(
            male_speakers, female_speakers, speaker_usage
        )
        interference_path = random.choice(interference_speaker_files[interference_speaker_id])
        interference_audio, _ = load_audio(interference_path, sr=sr)
        if fixed_length_sec is not None and fixed_length_sec > 0:
            length_samples = int(sr * fixed_length_sec)
            target_audio = adjust_audio_length(target_audio, length_samples)
            interference_audio = adjust_audio_length(interference_audio, length_samples)
        else:
            interference_audio = align_interference_to_target(target_audio, interference_audio)
        desired_snr = np.random.uniform(snr_min, snr_max)
        adjusted_interference = adjust_interference_for_snr(
            target_audio, interference_audio, desired_snr
        )
        mixed_audio = target_audio + adjusted_interference
        folder_path = os.path.join(output_split_dir, f"triplet{triplet_idx}")
        os.makedirs(folder_path, exist_ok=True)
        sf.write(os.path.join(folder_path, "target.wav"), target_audio, sr)
        sf.write(os.path.join(folder_path, "reference.wav"), reference_audio, sr)
        sf.write(os.path.join(folder_path, "mix.wav"), mixed_audio, sr)
        triplet_idx += 1


def build_train_val_interference_pools(voxceleb2_dir, metadata_csv, val_speaker_count):
    dev_files = list_wav_files(os.path.join(voxceleb2_dir, "dev"))
    dev_speaker_files = get_speaker_files(dev_files)
    dev_speaker_ids = list(dev_speaker_files.keys())
    val_speaker_ids = set(random.sample(dev_speaker_ids, val_speaker_count))
    train_speaker_ids = set(dev_speaker_ids) - val_speaker_ids
    male_dev, female_dev = load_gender_speakers(metadata_csv, "dev")
    train_male = [spk for spk in male_dev if spk in train_speaker_ids and spk in dev_speaker_files]
    train_female = [
        spk for spk in female_dev if spk in train_speaker_ids and spk in dev_speaker_files
    ]
    val_male = [spk for spk in male_dev if spk in val_speaker_ids and spk in dev_speaker_files]
    val_female = [spk for spk in female_dev if spk in val_speaker_ids and spk in dev_speaker_files]
    train_speaker_files = {
        spk: paths for spk, paths in dev_speaker_files.items() if spk in train_speaker_ids
    }
    val_speaker_files = {
        spk: paths for spk, paths in dev_speaker_files.items() if spk in val_speaker_ids
    }
    return train_speaker_files, train_male, train_female, val_speaker_files, val_male, val_female


def build_test_interference_pool(voxceleb2_dir, metadata_csv):
    test_files = list_wav_files(os.path.join(voxceleb2_dir, "test"))
    test_speaker_files = get_speaker_files(test_files)
    if not test_speaker_files:
        raise ValueError("no speaker found in voxceleb2 test split")
    male_test, female_test = load_gender_speakers(metadata_csv, "test")
    test_male = [spk for spk in male_test if spk in test_speaker_files]
    test_female = [spk for spk in female_test if spk in test_speaker_files]
    return test_speaker_files, test_male, test_female


def collect_libritts_targets(libritts_dir):
    train_dirs = [
        os.path.join(libritts_dir, "train-clean-100"),
        os.path.join(libritts_dir, "train-clean-360"),
    ]
    val_dir = os.path.join(libritts_dir, "dev-clean")
    test_dir = os.path.join(libritts_dir, "test-clean")
    train_files = []
    for directory in train_dirs:
        train_files.extend(list_wav_files(directory))
    val_files = list_wav_files(val_dir)
    test_files = list_wav_files(test_dir)
    return train_files, val_files, test_files


def generate_all_datasets(
    libritts_dir,
    voxceleb2_dir,
    output_dir,
    metadata_csv=None,
    snr_min=-5.0,
    snr_max=5.0,
    fixed_length_sec=6.0,
    val_speaker_count=94,
):
    metadata_csv = resolve_metadata_csv(metadata_csv, voxceleb2_dir)
    train_targets_raw, val_targets_raw, test_targets_raw = collect_libritts_targets(libritts_dir)
    train_targets, train_targets_by_speaker = filter_train_targets(train_targets_raw)
    val_targets, val_targets_by_speaker = filter_eval_targets(val_targets_raw)
    test_targets, test_targets_by_speaker = filter_eval_targets(test_targets_raw)
    (
        train_interference_files,
        train_male,
        train_female,
        val_interference_files,
        val_male,
        val_female,
    ) = build_train_val_interference_pools(voxceleb2_dir, metadata_csv, val_speaker_count)
    test_interference_files, test_male, test_female = build_test_interference_pool(
        voxceleb2_dir, metadata_csv
    )
    generate_split_triplets(
        target_files=train_targets,
        target_by_speaker_all=train_targets_by_speaker,
        interference_speaker_files=train_interference_files,
        male_speakers=train_male,
        female_speakers=train_female,
        output_split_dir=os.path.join(output_dir, "train"),
        snr_min=snr_min,
        snr_max=snr_max,
        split_name="train",
        fixed_length_sec=fixed_length_sec,
    )
    generate_split_triplets(
        target_files=val_targets,
        target_by_speaker_all=val_targets_by_speaker,
        interference_speaker_files=val_interference_files,
        male_speakers=val_male,
        female_speakers=val_female,
        output_split_dir=os.path.join(output_dir, "val"),
        snr_min=snr_min,
        snr_max=snr_max,
        split_name="val",
        fixed_length_sec=fixed_length_sec,
    )
    generate_split_triplets(
        target_files=test_targets,
        target_by_speaker_all=test_targets_by_speaker,
        interference_speaker_files=test_interference_files,
        male_speakers=test_male,
        female_speakers=test_female,
        output_split_dir=os.path.join(output_dir, "test"),
        snr_min=snr_min,
        snr_max=snr_max,
        split_name="test",
        fixed_length_sec=fixed_length_sec,
    )


def generate_test_dataset(
    libritts_dir,
    voxceleb2_dir,
    output_dir,
    metadata_csv=None,
    snr_min=-5.0,
    snr_max=5.0,
    fixed_length_sec=6.0,
):
    metadata_csv = resolve_metadata_csv(metadata_csv, voxceleb2_dir)
    test_targets_raw = list_wav_files(os.path.join(libritts_dir, "test-clean"))
    test_targets, test_targets_by_speaker = filter_eval_targets(test_targets_raw)
    test_interference_files, test_male, test_female = build_test_interference_pool(
        voxceleb2_dir, metadata_csv
    )
    generate_split_triplets(
        target_files=test_targets,
        target_by_speaker_all=test_targets_by_speaker,
        interference_speaker_files=test_interference_files,
        male_speakers=test_male,
        female_speakers=test_female,
        output_split_dir=output_dir,
        snr_min=snr_min,
        snr_max=snr_max,
        split_name="test",
        fixed_length_sec=fixed_length_sec,
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libritts_dir", required=True, help="Path to LibriTTS root.")
    parser.add_argument("--voxceleb2_dir", required=True, help="Path to VoxCeleb2 root.")
    parser.add_argument(
        "--output_dir",
        default="./output_dataset",
        help="Output directory that will contain train, val and test folders.",
    )
    parser.add_argument(
        "--metadata_csv",
        default=None,
        help="Path to vox2 metadata CSV. If omitted, common default paths are used.",
    )
    parser.add_argument("--snr_min", type=float, default=-5.0, help="Minimum SNR.")
    parser.add_argument("--snr_max", type=float, default=5.0, help="Maximum SNR.")
    parser.add_argument(
        "--fixed_length_sec",
        type=float,
        default=6.0,
        help="Fixed mix length in seconds. Set <=0 to disable fixed-length mode.",
    )
    parser.add_argument(
        "--val_speaker_count",
        type=int,
        default=94,
        help="Number of random dev speakers used for val interference.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.snr_min > args.snr_max:
        raise ValueError("snr_min must be <= snr_max")
    generate_all_datasets(
        libritts_dir=args.libritts_dir,
        voxceleb2_dir=args.voxceleb2_dir,
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        fixed_length_sec=args.fixed_length_sec,
        val_speaker_count=args.val_speaker_count,
    )


if __name__ == "__main__":
    main()
