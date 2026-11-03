from pathlib import Path
from audio.audio_utils import load_audio_for_gpu, log_audio_init_data

if __name__ == '__main__':
    print("audio_quality")
    BASE_DIR = Path(__file__).resolve().parent
    audio_file = BASE_DIR / "audio_samples" / "tehno.mp3"
    if not audio_file.exists():
        print(f"Error: file does not exists on path: {audio_file}")
    else:
        audio_data, sr = load_audio_for_gpu(str(audio_file))
        log_audio_init_data(audio_data, sr)