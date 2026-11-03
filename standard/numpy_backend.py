import numpy as np
import os
import scipy.signal
import soundfile as sf


def analyze(batch_pcm: np.ndarray, lengths: list[int], cfg: dict):
    """
    Gold standard for audio analysis on cpu.
    batch_pcm: int16 matrix shape (batch_size, channels, max_samples)
    lengths: number of valid samples for every file
    cfg: configuration parameters
    """
    batch_size = batch_pcm.shape[0]
    batch_float = batch_pcm.astype(np.float32) / 32768.0

    if batch_float.shape[1] != 2:
        raise ValueError("Backend expecting stereo (2 channels)")

    L = batch_float[:, 0, :]
    R = batch_float[:, 1, :]

    M = (L + R) / 2.0
    S = (L - R) / 2.0
    batch_ms = np.stack((M, S), axis=1)

    n_fft = cfg.get("n_fft", 4096)
    hop_length = cfg.get("hop_length", 2048)
    sample_rate = cfg.get("sample_rate", 44100)

    win = scipy.signal.windows.hann(n_fft, sym=False)

    freqs, times, Zxx = scipy.signal.stft(
        batch_ms,
        fs=sample_rate,
        window=win,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        return_onesided=True,
        boundary=None,
        padded=False
    )

    power = np.abs(Zxx) ** 2
    db_spectrogram = 10.0 * np.log10(power + 1e-10)

    n_freqs = Zxx.shape[2]
    ltas = np.zeros((batch_size, 2, n_freqs), dtype=np.float32)
    percentile_90 = np.zeros((batch_size, 2, n_freqs), dtype=np.float32)
    dc_offset = np.zeros((batch_size, 2), dtype=np.float32)
    rms = np.zeros((batch_size, 2), dtype=np.float32)
    true_peak = np.zeros((batch_size, 2), dtype=np.float32)
    clipped_samples = np.zeros((batch_size, 2), dtype=np.int64)
    stereo_correlation = np.zeros((batch_size,), dtype=np.float32)

    cutoff_bin = int(16000 / (sample_rate / n_fft))
    hf_energy_ratio = np.zeros((batch_size, 2), dtype=np.float32)

    for i in range(batch_size):
        valid_len = lengths[i]

        valid_pcm = batch_ms[i, :, :valid_len]
        valid_L = L[i, :valid_len]
        valid_R = R[i, :valid_len]

        dc_offset[i] = np.mean(valid_pcm, axis=-1)
        rms[i] = np.sqrt(np.mean(valid_pcm ** 2, axis=-1))
        true_peak[i] = np.max(np.abs(valid_pcm), axis=-1)

        clipped_samples[i] = np.sum(np.abs(valid_pcm) >= 0.9999, axis=-1)

        nom = np.sum(valid_L * valid_R)
        denom = np.sqrt(np.sum(valid_L ** 2) * np.sum(valid_R ** 2)) + 1e-10
        stereo_correlation[i] = nom / denom

        if valid_len >= n_fft:
            valid_frames = (valid_len - n_fft) // hop_length + 1
        else:
            valid_frames = 0

        valid_db = db_spectrogram[i, :, :, :valid_frames]
        valid_power = power[i, :, :, :valid_frames]

        ltas[i] = np.mean(valid_db, axis=-1)
        percentile_90[i] = np.percentile(valid_db, 90, axis=-1)

        total_energy = np.sum(valid_power, axis=(1, 2)) + 1e-10
        hf_energy = np.sum(valid_power[:, cutoff_bin:, :], axis=(1, 2))
        hf_energy_ratio[i] = hf_energy / total_energy

    return {
        "ltas": ltas,
        "percentile_90": percentile_90,
        "dc_offset": dc_offset,
        "rms": rms,
        "true_peak": true_peak,
        "clipped_samples": clipped_samples,
        "stereo_correlation": stereo_correlation,
        "hf_energy_ratio": hf_energy_ratio,
        "freqs": freqs,
        "times": times
    }


def load_audio_batch(folder_path: str, filenames: list[str]) -> tuple[np.ndarray, list[int]]:
    """
    Loading FLAC/WAV files, assuring stereo format and setting zeros
    so all files in batch would be the same length
    """
    audios = []
    lengths = []

    for fname in filenames:
        path = os.path.join(folder_path, fname)
        data, samplerate = sf.read(path, dtype='int16')

        if data.ndim == 1:
            data = np.stack((data, data), axis=1)

        data = data.T
        audios.append(data)
        lengths.append(data.shape[1])

    max_len = max(lengths)
    batch_size = len(audios)

    batch_pcm = np.zeros((batch_size, 2, max_len), dtype=np.int16)

    for i, data in enumerate(audios):
        batch_pcm[i, :, :data.shape[1]] = data

    return batch_pcm, lengths