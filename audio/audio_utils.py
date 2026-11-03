import numpy as np
import soundfile as sf


def load_audio_for_gpu(file_path: str):
    data, samplerate = sf.read(file_path, dtype='float32')
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data, dtype=np.float32)
    return data, samplerate

def log_audio_init_data(audio_data, sr):
    print("File successfully loaded!!!")
    print(f"Sample rate: {sr} HZ")
    print(f"Total samples: {len(audio_data)}")
    print(f"Length: {len(audio_data) / sr:.2f} seconds")
    print(f"Address in a ram memory (pointer): {audio_data.ctypes.data}")