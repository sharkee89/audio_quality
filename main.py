from standard.numpy_backend import analyze, load_audio_batch

if __name__ == "__main__":
    folder_data = "./audio_samples"

    test_files = ["tehno.mp3"]

    cfg = {
        "n_fft": 4096,
        "hop_length": 2048,
        "sample_rate": 44100
    }

    print("Loading files...")
    batch_pcm, lengths = load_audio_batch(folder_data, test_files)

    print(f"Successfully loaded batch with shape: {batch_pcm.shape}")
    print("Initiating audio analysis...")

    res = analyze(batch_pcm, lengths, cfg)

    print("\nFinished! results data:")
    print(f"LTAS matrix shape: {res['ltas'].shape}")
    print(f"RMS song values: {res['rms']}")