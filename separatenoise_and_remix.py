from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch as th
import torch.nn as nn
from tqdm import tqdm


def hz2mel(hz: float) -> float:
    return 2595 * np.log10(1 + hz / 700.0)


def mel2hz(mel: float) -> float:
    return 700 * (10 ** (mel / 2595.0) - 1)


def hz2bark(hz: float) -> float:
    return 6.0 * np.arcsinh(hz / 600.0)


def bark2hz(bark: float) -> float:
    return 600.0 * np.sinh(bark / 6.0)


def hz2erb(hz: float) -> float:
    return 21.4 * np.log10(1 + 0.00437 * hz)


def erb2hz(erb: float) -> float:
    return (np.power(10, (erb / 21.4)) - 1) / 0.00437


def filter_banks(
    filter_type: str, hz_low: float, hz_high: float, hz_bands_num: int, filters_num: int
) -> Tuple[np.ndarray, np.ndarray]:
    if filter_type == "mel":
        hz2filter = hz2mel
        filter2hz = mel2hz
    elif filter_type == "bark":
        hz2filter = hz2bark
        filter2hz = bark2hz
    elif filter_type == "erb":
        hz2filter = hz2erb
        filter2hz = erb2hz
    else:
        raise NameError("filter_type must be mel, bark or erb")

    filter_low = hz2filter(hz_low)
    filter_high = hz2filter(hz_high)
    filter_linear = np.linspace(filter_low, filter_high, filters_num + 2)
    filter_hz = filter2hz(filter_linear)
    min_filter_hz = filter_hz[1]
    hz_band_width = hz_high / (hz_bands_num - 1)
    if min_filter_hz < hz_band_width:
        raise NameError("too many filters, decrease filters number")

    filter_matrix = np.zeros((hz_bands_num, filters_num), dtype=np.float32)
    for filter_idx in range(1, filters_num + 1):
        low = np.floor(filter_hz[filter_idx - 1] / hz_band_width)
        mid = np.floor(filter_hz[filter_idx] / hz_band_width)
        hig = np.floor(filter_hz[filter_idx + 1] / hz_band_width)
        for hz_idx in range(hz_bands_num):
            if low <= hz_idx <= mid:
                filter_matrix[hz_idx, filter_idx - 1] = (hz_idx - low) / (mid - low)
            elif mid < hz_idx <= hig:
                if filter_idx < filters_num:
                    filter_matrix[hz_idx, filter_idx - 1] = (hig - hz_idx) / (hig - mid)
                else:
                    filter_matrix[hz_idx, filter_idx - 1] = 1
    filter_scale = filter_matrix.sum(0)
    return filter_matrix, filter_scale


class LocalDenseConv1D(nn.Module):
    def __init__(
        self,
        in_c: int,
        in_l: int,
        out_c: int,
        out_l: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        bn_norm: bool = True,
        activation: nn.Module = nn.PReLU(),
    ):
        super().__init__()
        self.in_c = in_c
        self.in_l = in_l
        self.out_c = out_c
        self.out_l = out_l
        self.out_l_unfold = (in_l + padding * 2 - kernel) // stride + 1
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.unfold = nn.Unfold(
            kernel_size=(self.kernel, 1), stride=(self.stride, 1), padding=(self.padding, 0)
        )
        self.weight = nn.Parameter(
            torch.Tensor(1, self.in_c * self.kernel, self.out_c, self.out_l, 1)
        )
        nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, self.out_c, self.out_l, 1))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None
        if bn_norm:
            self.norm = nn.BatchNorm2d(num_features=self.out_c)
        else:
            self.norm = None
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_c, _, t_len = x.shape
        x_unfold = self.unfold(x)
        x_unfold = x_unfold.view(batch_size, in_c * self.kernel, self.out_l_unfold, t_len)
        x_unfold = x_unfold[:, :, : self.out_l, :].unsqueeze(2)
        out = (x_unfold * self.weight).sum(1)
        if self.bias is not None:
            out = out + self.bias
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


class LocalDenseConv2D(nn.Module):
    def __init__(
        self,
        in_c: int,
        in_l: int,
        out_c: int,
        out_l: int,
        kernel: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        bias: bool = True,
        bn_norm: bool = True,
        activation: nn.Module = nn.PReLU(),
    ):
        super().__init__()
        self.in_c = in_c
        self.in_l = in_l
        self.out_c = out_c
        self.out_l = out_l
        self.out_l_unfold = (in_l + padding[0] * 2 - kernel[0]) // stride[0] + 1
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.unfold = nn.Unfold(kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.weight = nn.Parameter(
            torch.Tensor(1, self.in_c * self.kernel[0] * self.kernel[1], self.out_c, self.out_l, 1)
        )
        nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, self.out_c, self.out_l, 1))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None
        if bn_norm:
            self.norm = nn.BatchNorm2d(num_features=self.out_c)
        else:
            self.norm = None
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_c, _, t_len = x.shape
        out_t = (t_len + self.padding[1] * 2 - self.kernel[1]) // self.stride[1] + 1
        x_unfold = self.unfold(x)
        x_unfold = x_unfold.view(
            batch_size, in_c * self.kernel[0] * self.kernel[1], self.out_l_unfold, out_t
        )
        x_unfold = x_unfold[:, :, : self.out_l, :t_len].unsqueeze(2)
        out = (x_unfold * self.weight).sum(1)
        if self.bias is not None:
            out = out + self.bias
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


class GroupedLinear(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, groups: int = 1, shuffle: bool = True):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.groups = groups
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        if groups == 1:
            shuffle = False
        self.shuffle = shuffle
        self.layers = nn.ModuleList(
            nn.Linear(self.input_size, self.hidden_size) for _ in range(groups)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, layer in enumerate(self.layers):
            outputs.append(layer(x[..., i * self.input_size : (i + 1) * self.input_size]))
        output = torch.cat(outputs, dim=-1)
        if self.shuffle:
            orig_shape = output.shape
            output = (
                output.view(orig_shape[0], orig_shape[1], self.groups, self.hidden_size)
                .transpose(-1, -2)
                .reshape(orig_shape)
            )
        return output


class SqueezedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
        batch_first: bool = True,
        gru_skip_op: type = nn.Identity,
        linear_act_layer: type = nn.Identity,
    ):
        super().__init__()
        self.linear_in = nn.Sequential(
            GroupedLinear(input_size, hidden_size, linear_groups), linear_act_layer()
        )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                GroupedLinear(hidden_size, output_size, linear_groups), linear_act_layer()
            )
        else:
            self.linear_out = nn.Identity()

    def forward(self, input_x: torch.Tensor, h: Optional[torch.Tensor] = None):
        input_x = self.linear_in(input_x)
        x, h = self.gru(input_x, h)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input_x)
        x = self.linear_out(x)
        return x, h


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.erb_norm = nn.BatchNorm2d(1)
        self.conv_erb_1 = LocalDenseConv2D(
            in_c=1, in_l=36, out_c=8, out_l=36, kernel=(2, 2), stride=(1, 1), padding=(1, 1)
        )
        self.conv_erb_2 = LocalDenseConv1D(
            in_c=8, in_l=36, out_c=12, out_l=18, kernel=4, stride=2, padding=2
        )
        self.conv_erb_3 = LocalDenseConv1D(
            in_c=12, in_l=18, out_c=15, out_l=6, kernel=6, stride=3, padding=3
        )

        self.com_norm = nn.BatchNorm2d(2)
        self.conv_com_1 = LocalDenseConv2D(
            in_c=2, in_l=320, out_c=4, out_l=80, kernel=(4, 2), stride=(4, 1), padding=(2, 1)
        )
        self.conv_com_2 = LocalDenseConv1D(
            in_c=4, in_l=80, out_c=8, out_l=40, kernel=4, stride=2, padding=2
        )
        self.conv_com_3 = LocalDenseConv1D(
            in_c=8, in_l=40, out_c=9, out_l=10, kernel=4, stride=4, padding=2
        )

        self.gru_cat = SqueezedGRU(180, 48, 80, linear_groups=2, linear_act_layer=nn.PReLU)

    def forward(self, erb_feat: torch.Tensor, com_feat: torch.Tensor):
        erb_e0 = self.erb_norm(erb_feat)
        erb_e1 = self.conv_erb_1(erb_e0)
        erb_e2 = self.conv_erb_2(erb_e1)
        erb_e3 = self.conv_erb_3(erb_e2)

        com_e0 = self.com_norm(com_feat)
        com_e1 = self.conv_com_1(com_e0)
        com_e2 = self.conv_com_2(com_e1)
        com_e3 = self.conv_com_3(com_e2)
        com_e3 = com_e3.permute(0, 3, 1, 2).flatten(2)

        erb_e3_cat = erb_e3.permute(0, 3, 1, 2).flatten(2)
        cat_feat = torch.cat((erb_e3_cat, com_e3), -1)
        cat_gru, _ = self.gru_cat(cat_feat)
        return cat_gru, erb_e1, erb_e2, erb_e3


class ErbDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = SqueezedGRU(80, 48, 90, linear_groups=2, linear_act_layer=nn.PReLU)

        self.pconv3 = LocalDenseConv1D(
            in_c=15, in_l=6, out_c=15, out_l=6, kernel=1, stride=1, padding=0, bias=False, bn_norm=False
        )
        self.pconv2 = LocalDenseConv1D(
            in_c=12, in_l=18, out_c=12, out_l=18, kernel=1, stride=1, padding=0, bias=False, bn_norm=False
        )
        self.pconv1 = LocalDenseConv1D(
            in_c=8, in_l=36, out_c=8, out_l=36, kernel=1, stride=1, padding=0, bias=False, bn_norm=False
        )

        self.conv3 = LocalDenseConv1D(
            in_c=15, in_l=6, out_c=36, out_l=6, kernel=2, stride=1, padding=1
        )
        self.conv2 = LocalDenseConv1D(
            in_c=12, in_l=18, out_c=16, out_l=18, kernel=2, stride=1, padding=1
        )
        self.conv1 = LocalDenseConv2D(
            in_c=8,
            in_l=36,
            out_c=1,
            out_l=36,
            kernel=(4, 2),
            stride=(1, 1),
            padding=(2, 1),
            activation=nn.Sigmoid(),
        )

    def forward(
        self, cat_gru: torch.Tensor, erb_e1: torch.Tensor, erb_e2: torch.Tensor, erb_e3: torch.Tensor
    ) -> torch.Tensor:
        gru_out, _ = self.gru(cat_gru)
        b_size, ch, freq, t_len = erb_e3.shape
        gru_out = gru_out.reshape(b_size, t_len, ch, freq).permute(0, 2, 3, 1)
        d3 = self.conv3(self.pconv3(erb_e3) + gru_out)
        d3 = d3.contiguous().view(erb_e2.shape)
        d2 = self.conv2(self.pconv2(erb_e2) + d3)
        d2 = d2.contiguous().view(erb_e1.shape)
        d1 = self.conv1(self.pconv1(erb_e1) + d2)
        return d1


class ComDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru_speech = SqueezedGRU(80, 64, 160, linear_groups=2, linear_act_layer=nn.PReLU)
        self.gru_noise = SqueezedGRU(80, 64, 160, linear_groups=2, linear_act_layer=nn.PReLU)

        self.conv_speech = LocalDenseConv2D(
            in_c=2,
            in_l=160,
            out_c=1,
            out_l=160,
            kernel=(4, 2),
            stride=(1, 1),
            padding=(2, 1),
            activation=nn.Sigmoid(),
        )
        self.conv_noise = LocalDenseConv2D(
            in_c=2,
            in_l=160,
            out_c=1,
            out_l=160,
            kernel=(4, 2),
            stride=(1, 1),
            padding=(2, 1),
            activation=nn.Sigmoid(),
        )

        self.linear_speech = nn.Sequential(GroupedLinear(160, 320, 5, shuffle=False), nn.Sigmoid())
        self.linear_noise = nn.Sequential(GroupedLinear(160, 320, 5, shuffle=False), nn.Sigmoid())

    def forward(self, cat_gru: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gru_out_s, _ = self.gru_speech(cat_gru)
        gru_out_n, _ = self.gru_noise(cat_gru)
        cat_feat = torch.stack((gru_out_s, gru_out_n), 1)
        cat_feat = cat_feat.permute(0, 1, 3, 2)

        mask_s = self.conv_speech(cat_feat).squeeze(1)
        mask_s = mask_s.squeeze(1).permute(0, 2, 1)
        mask_n = self.conv_noise(cat_feat).squeeze(1)
        mask_n = mask_n.squeeze(1).permute(0, 2, 1)

        feat_s = gru_out_s + gru_out_n * mask_s
        feat_n = gru_out_n + gru_out_s * mask_n

        lin_out_s = self.linear_speech(feat_s)
        output_mask_s = lin_out_s.permute(0, 2, 1)
        lin_out_n = self.linear_noise(feat_n)
        output_mask_n = lin_out_n.permute(0, 2, 1)
        return output_mask_s, output_mask_n


class NET(nn.Module):
    def __init__(self):
        super().__init__()
        erb_filter, filter_scale = filter_banks("mel", 0, 8000, 321, 36)
        self.erb_filter = torch.FloatTensor(erb_filter)
        self.filter_scale = torch.FloatTensor(filter_scale)
        self.encoder = Encoder()
        self.erb_decoder = ErbDecoder()
        self.com_decoder = ComDecoder()

    def forward(self, complex_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        erb_feat, com_feat = self.feature_extract(complex_input)
        encoder_feat, erb_e1, erb_e2, erb_e3 = self.encoder(erb_feat, com_feat)
        erb_gain = self.erb_decoder(encoder_feat, erb_e1, erb_e2, erb_e3)
        erb_output = self.erb_gain_apply(complex_input, erb_gain)

        speech_mask, noise_mask = self.com_decoder(encoder_feat)
        complex_output = self.com_gain_apply(complex_input, speech_mask)
        noise_output = self.com_gain_apply(complex_input, noise_mask)
        return erb_output, noise_output, complex_output

    def feature_extract(self, complex_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mag_erb = torch.matmul(
            complex_input.abs().pow(2).permute(0, 2, 1), self.erb_filter
        ) / self.filter_scale
        erb_feat = torch.log10(mag_erb + 1e-8).permute(0, 2, 1).unsqueeze(1)
        com_feat = torch.stack((complex_input[:, 1:, :].real, complex_input[:, 1:, :].imag), axis=1).float()
        return erb_feat, com_feat

    def erb_gain_apply(self, complex_input: torch.Tensor, erb_gain: torch.Tensor) -> torch.Tensor:
        erb_gain_recover = torch.matmul(
            erb_gain.squeeze(1).permute(0, 2, 1), self.erb_filter.permute(1, 0)
        ).permute(0, 2, 1)
        erb_output = erb_gain_recover * complex_input
        return erb_output

    def com_gain_apply(self, complex_input: torch.Tensor, com_mask: torch.Tensor) -> torch.Tensor:
        complex_output = torch.zeros_like(complex_input)
        complex_output[:, 1:, :] = complex_input[:, 1:, :] * com_mask
        return complex_output


def compute_amplitude(waveform: np.ndarray) -> float:
    return float(np.mean(np.abs(waveform)))


def align_to_length(waveform: np.ndarray, target_length: int) -> np.ndarray:
    if len(waveform) == target_length:
        return waveform
    if len(waveform) > target_length:
        return waveform[:target_length]
    return np.pad(waveform, (0, target_length - len(waveform)), mode="constant")


def parse_extensions(raw_ext: str) -> Tuple[str, ...]:
    exts = []
    for item in raw_ext.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = f".{item}"
        exts.append(item)
    if not exts:
        raise ValueError("No valid audio extension in --extensions.")
    return tuple(sorted(set(exts)))


def list_audio_files(audio_dir: Path, extensions: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for path in audio_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)
    files.sort()
    return files


def load_audio_mono(audio_path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def resolve_speech_path(
    speech_dir: Path, rel_path: Path, extensions: Tuple[str, ...]
) -> Optional[Path]:
    exact = speech_dir / rel_path
    if exact.exists():
        return exact

    stem = rel_path.stem
    parent = speech_dir / rel_path.parent
    for ext in extensions:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


class NoiseSeparatorRemixer:
    def __init__(
        self,
        model_weights: Path,
        sample_rate: int = 16000,
        n_fft: int = 640,
        hop_length: int = 320,
        device: str = "auto",
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.device = self._resolve_device(device)

        half_win = np.sin(
            0.5
            * np.pi
            * (
                np.sin(np.pi / self.n_fft * np.linspace(0.5, self.hop_length - 0.5, self.hop_length))
                ** 2
            )
        )
        full_win = np.hstack((half_win, half_win[::-1])).astype(np.float32)
        self.window = th.from_numpy(full_win).to(self.device)

        self.model = NET()
        self._load_weights(model_weights)
        self.model = self.model.to(self.device)
        self.model.erb_filter = self.model.erb_filter.to(self.device)
        self.model.filter_scale = self.model.filter_scale.to(self.device)
        self.model.eval()

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_weights(self, model_weights: Path):
        if not model_weights.is_file():
            raise FileNotFoundError(f"Model weights not found: {model_weights}")
        checkpoint = th.load(str(model_weights), map_location="cpu")
        if isinstance(checkpoint, dict):
            if "net" in checkpoint and isinstance(checkpoint["net"], dict):
                state_dict = checkpoint["net"]
            elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            raise ValueError("Unsupported checkpoint format.")

        cleaned_state_dict: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            cleaned_state_dict[key.replace("module.", "")] = value
        missing, unexpected = self.model.load_state_dict(cleaned_state_dict, strict=False)
        if missing:
            print(f"[Warn] Missing keys when loading checkpoint: {len(missing)}")
        if unexpected:
            print(f"[Warn] Unexpected keys when loading checkpoint: {len(unexpected)}")

    @torch.inference_mode()
    def separate(self, noisy_audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mixture = th.from_numpy(noisy_audio).float().unsqueeze(0).to(self.device)
        stft_mix = th.stft(
            mixture,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        _, stft_noise_est, stft_clean_est = self.model(stft_mix)

        noise_est = th.istft(
            stft_noise_est,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=mixture.shape[-1],
        )
        clean_est = th.istft(
            stft_clean_est,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=mixture.shape[-1],
        )
        noise_np = noise_est.squeeze(0).detach().cpu().numpy()
        clean_np = clean_est.squeeze(0).detach().cpu().numpy()
        return noise_np.astype(np.float32), clean_np.astype(np.float32)


def process_folder(
    engine: NoiseSeparatorRemixer,
    input_dir: Path,
    output_dir: Path,
    speech_dir: Optional[Path],
    save_noise_dir: Optional[Path],
    output_suffix: str,
    extensions: Tuple[str, ...],
    clip_audio: bool,
    overwrite: bool,
):
    audio_files = list_audio_files(input_dir, extensions)
    if not audio_files:
        raise ValueError(f"No audio file found in {input_dir} with extensions {extensions}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_noise_dir is not None:
        save_noise_dir.mkdir(parents=True, exist_ok=True)

    for input_path in tqdm(audio_files, desc="Processing"):
        rel_path = input_path.relative_to(input_dir)
        output_parent = output_dir / rel_path.parent
        output_parent.mkdir(parents=True, exist_ok=True)

        out_name = f"{rel_path.stem}{output_suffix}.wav"
        output_path = output_parent / out_name
        if output_path.exists() and not overwrite:
            continue

        noisy_audio = load_audio_mono(input_path, engine.sample_rate)
        noise_est, clean_est = engine.separate(noisy_audio)

        if speech_dir is not None:
            speech_path = resolve_speech_path(speech_dir, rel_path, extensions)
            if speech_path is None:
                print(f"[Warn] Missing speech file for {rel_path}, fallback to model clean estimate.")
                speech_audio = clean_est
            else:
                speech_audio = load_audio_mono(speech_path, engine.sample_rate)
        else:
            speech_audio = clean_est

        speech_audio = align_to_length(speech_audio, len(noise_est))
        noisy_aligned = align_to_length(noisy_audio, len(noise_est))
        ori_amp = compute_amplitude(noisy_aligned)
        speech_amp = compute_amplitude(speech_audio)
        scale = ori_amp / max(speech_amp, 1e-8)

        remixed = speech_audio * scale + noise_est
        if clip_audio:
            remixed = np.clip(remixed, -1.0, 1.0)
        sf.write(str(output_path), remixed.astype(np.float32), engine.sample_rate)

        if save_noise_dir is not None:
            noise_parent = save_noise_dir / rel_path.parent
            noise_parent.mkdir(parents=True, exist_ok=True)
            noise_path = noise_parent / f"{rel_path.stem}_noise.wav"
            sf.write(str(noise_path), noise_est.astype(np.float32), engine.sample_rate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Separate noise with NET and remix with optional speech folder."
    )
    parser.add_argument("--input_dir", required=True, help="Input noisy audio folder.")
    parser.add_argument("--output_dir", required=True, help="Output remixed audio folder.")
    parser.add_argument("--weights", required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--speech_dir",
        default=None,
        help="Optional speech folder (same relative paths as input_dir). "
        "If omitted, use model estimated clean speech.",
    )
    parser.add_argument(
        "--save_noise_dir",
        default=None,
        help="Optional folder to save estimated noise wav files.",
    )
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sampling rate.")
    parser.add_argument("--n_fft", type=int, default=640, help="STFT n_fft.")
    parser.add_argument("--hop_length", type=int, default=320, help="STFT hop_length.")
    parser.add_argument(
        "--device",
        default="auto",
        help='Inference device: "auto", "cpu", "cuda", or specific like "cuda:0".',
    )
    parser.add_argument(
        "--extensions",
        default=".wav,.flac,.mp3,.m4a,.ogg",
        help='Comma separated audio extensions, e.g. ".wav,.flac".',
    )
    parser.add_argument(
        "--output_suffix",
        default="_eh",
        help="Suffix for remixed filename before .wav.",
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="Disable clipping to [-1, 1] before saving.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    speech_dir = Path(args.speech_dir).expanduser().resolve() if args.speech_dir else None
    save_noise_dir = (
        Path(args.save_noise_dir).expanduser().resolve() if args.save_noise_dir else None
    )

    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir not found: {input_dir}")
    if speech_dir is not None and not speech_dir.is_dir():
        raise NotADirectoryError(f"speech_dir not found: {speech_dir}")

    extensions = parse_extensions(args.extensions)
    clip_audio = not args.no_clip

    engine = NoiseSeparatorRemixer(
        model_weights=weights,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        device=args.device,
    )
    process_folder(
        engine=engine,
        input_dir=input_dir,
        output_dir=output_dir,
        speech_dir=speech_dir,
        save_noise_dir=save_noise_dir,
        output_suffix=args.output_suffix,
        extensions=extensions,
        clip_audio=clip_audio,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
