import numpy as np
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

    print("\n--- Audio Analysis & Quality Metrics ---")
    for i, fname in enumerate(test_files):
        print(f"\nFile: {fname} (Valid samples: {lengths[i]})")
        print(f"  RMS (Mid / Side):           {res['rms'][i]}")
        print(f"  True Peak (Mid / Side):     {res['true_peak'][i]}")
        print(f"  DC Offset (Mid / Side):     {res['dc_offset'][i]}")
        print(f"  Clipped Samples:            {res['clipped_samples'][i]}")
        print(f"  Stereo Phase Correlation:   {res['stereo_correlation'][i]:.4f}")
        print(f"  HF Energy Ratio (>16kHz):   {res['hf_energy_ratio'][i]}")

    print(f"\nLTAS matrix shape: {res['ltas'].shape}")
    print(f"Percentile 90 shape: {res['percentile_90'].shape}")

    # Snimanje referentnih vrednosti za verifikaciju PyTorch / CUDA koda
    output_npz = "ground_truth_test.npz"
    np.savez_compressed(
        output_npz,
        batch_pcm=batch_pcm,
        lengths=np.array(lengths),
        ltas=res["ltas"],
        percentile_90=res["percentile_90"],
        dc_offset=res["dc_offset"],
        rms=res["rms"],
        true_peak=res["true_peak"],
        clipped_samples=res["clipped_samples"],
        stereo_correlation=res["stereo_correlation"],
        hf_energy_ratio=res["hf_energy_ratio"]
    )
    print(f"\nSuccessfully saved ground truth to '{output_npz}'.")