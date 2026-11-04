import torch
import numpy as np

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    else:
        return torch.device("cpu")

def analyze_torch(batch_pcm: np.ndarray, lengths: list[int], cfg: dict, device=None):
    if device is None:
        device = get_device()

    batch_tensor = torch.from_numpy(batch_pcm).to(device)
    batch_size = batch_tensor.shape[0]
    batch_float = batch_tensor.to(torch.float32) / 32768.0

    if batch_float.shape[1] != 2:
        raise ValueError("Backend expecting stereo (2 channels)")

    L = batch_float[:, 0, :]
    R = batch_float[:, 1, :]

    M = (L + R) / 2.0
    S = (L - R) / 2.0

    batch_ms = torch.stack((M, S), dim=1)

    n_fft = cfg.get("n_fft", 4096)
    hop_length = cfg.get("hop_length", 2048)
    sample_rate = cfg.get("sample_rate", 44100)
    window = torch.hann_window(n_fft, periodic=True, device=device)

    orig_shape = batch_ms.shape
    flat_ms = batch_ms.reshape(-1, orig_shape[-1])

    Zxx = torch.stft(
        flat_ms,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True
    )

    Zxx = Zxx.reshape(orig_shape[0], orig_shape[1], Zxx.shape[-2], Zxx.shape[-1])
    win_sum = window.sum()
    Zxx = Zxx / win_sum

    power = torch.abs(Zxx) ** 2
    db_spectrogram = 10.0 * torch.log10(power + 1e-10)

    n_freqs = Zxx.shape[2]
    ltas = torch.zeros((batch_size, 2, n_freqs), dtype=torch.float32, device=device)
    percentile_90 = torch.zeros((batch_size, 2, n_freqs), dtype=torch.float32, device=device)
    dc_offset = torch.zeros((batch_size, 2), dtype=torch.float32, device=device)
    rms = torch.zeros((batch_size, 2), dtype=torch.float32, device=device)
    sample_peak = torch.zeros((batch_size, 2), dtype=torch.float32, device=device)
    clipped_samples = torch.zeros((batch_size, 2), dtype=torch.int64, device=device)
    stereo_correlation = torch.zeros((batch_size,), dtype=torch.float32, device=device)

    cutoff_bin = int(16000 / (sample_rate / n_fft))
    hf_energy_ratio = torch.zeros((batch_size, 2), dtype=torch.float32, device=device)

    for i in range(batch_size):
        valid_len = lengths[i]

        valid_pcm = batch_ms[i, :, :valid_len]
        valid_L = L[i, :valid_len]
        valid_R = R[i, :valid_len]

        dc_offset[i] = torch.mean(valid_pcm, dim=-1)
        rms[i] = torch.sqrt(torch.mean(valid_pcm ** 2, dim=-1))
        sample_peak[i] = torch.max(torch.abs(valid_pcm), dim=-1).values

        valid_lr = torch.stack((valid_L, valid_R), dim=0)
        clipped_samples[i] = torch.sum((torch.abs(valid_lr) >= 0.9999).to(torch.int64), dim=-1)

        nom = torch.sum(valid_L * valid_R)
        denom = torch.sqrt(torch.sum(valid_L ** 2) * torch.sum(valid_R ** 2)) + 1e-10
        stereo_correlation[i] = nom / denom

        if valid_len >= n_fft:
            valid_frames = (valid_len - n_fft) // hop_length + 1
        else:
            valid_frames = 0

        valid_db = db_spectrogram[i, :, :, :valid_frames]
        valid_power = power[i, :, :, :valid_frames]

        ltas[i] = torch.mean(valid_db, dim=-1)

        percentile_90[i] = torch.quantile(valid_db, q=0.90, dim=-1)

        total_energy = torch.sum(valid_power, dim=(1, 2)) + 1e-10
        hf_energy = torch.sum(valid_power[:, cutoff_bin:, :], dim=(1, 2))
        hf_energy_ratio[i] = hf_energy / total_energy

    return {
        "ltas": ltas.cpu().numpy(),
        "percentile_90": percentile_90.cpu().numpy(),
        "dc_offset": dc_offset.cpu().numpy(),
        "rms": rms.cpu().numpy(),
        "sample_peak": sample_peak.cpu().numpy(),
        "clipped_samples": clipped_samples.cpu().numpy(),
        "stereo_correlation": stereo_correlation.cpu().numpy(),
        "hf_energy_ratio": hf_energy_ratio.cpu().numpy()
    }