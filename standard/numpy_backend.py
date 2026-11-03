import numpy as np
import os
import scipy.signal
import soundfile as sf

def analyze(batch_pcm: np.ndarray, lengths: list[int], cfg: dict):
    """
    Gold standard for audio analysis on cpu.
    batch_pcm: int16 matrix shape (batch_size, channels, max_samples)
    lengths: number of samples for every file (because of batch padding)
    cfg: configuration parameters
    """
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

    freqs, times, Zxx = scipy.signal.stft(
        batch_ms,
        fs=sample_rate,
        window='hann',
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        return_onesided=True,
        boundary=None,
        padded=False
    )


    power = np.abs(Zxx) ** 2
    db_spectrogram = 10 * np.log10(power + 1e-10)

    ltas = np.mean(db_spectrogram, axis=-1)
    percentile_90 = np.percentile(db_spectrogram, 90, axis=-1)
    dc_offset = np.mean(batch_ms, axis=-1)
    rms = np.sqrt(np.mean(batch_ms ** 2, axis=-1))
    true_peak = np.max(np.abs(batch_ms), axis=-1)

    return {
        "ltas": ltas,
        "percentile_90": percentile_90,
        "dc_offset": dc_offset,
        "rms": rms,
        "true_peak": true_peak
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