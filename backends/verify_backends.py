import numpy as np
import os
import sys
from backends.torch_backend import analyze_torch, get_device

def run_verification(npz_path="../ground_truth_test.npz"):
    if not os.path.exists(npz_path):
        print(f"Greska: Fajl '{npz_path}' ne postoji. Prvo pokreni main.py sa numpy backendom!")
        sys.exit(1)

    print(f"Loading file: {npz_path}...")
    ref = np.load(npz_path)

    batch_pcm = ref["batch_pcm"]
    lengths = ref["lengths"].tolist()

    cfg = {
        "n_fft": 4096,
        "hop_length": 2048,
        "sample_rate": 44100
    }

    device = get_device()
    print(f"Executing PyTorch analysis on the device: {device}...")
    torch_res = analyze_torch(batch_pcm, lengths, cfg, device=device)

    # Definisanje metrika za proveru i dozvoljenih numerickih tolerancija
    # atol (apsolutna tolerancija) i rtol (relativna tolerancija)
    checks = [
        ("RMS", "rms", 1e-5, 1e-6),
        ("True Peak", "true_peak", 1e-5, 1e-6),
        ("DC Offset", "dc_offset", 1e-5, 1e-6),
        ("Clipped Samples", "clipped_samples", 0, 0),  # celi brojevi moraju biti 100% identicni
        ("Stereo Correlation", "stereo_correlation", 1e-4, 1e-5),
        ("HF Energy Ratio", "hf_energy_ratio", 1e-4, 1e-5),
        ("LTAS Matrix", "ltas", 1e-4, 1e-4),
        ("Percentile 90", "percentile_90", 1e-3, 1e-3),
    ]

    all_passed = True
    print("\n--- Sanity Check ---")

    for label, key, rtol, atol in checks:
        ref_val = ref[key]
        t_val = torch_res[key]

        try:
            if key == "clipped_samples":
                np.testing.assert_array_equal(t_val, ref_val)
            else:
                np.testing.assert_allclose(t_val, ref_val, rtol=rtol, atol=atol)
            print(f"[PASS] {label:<22} successfully!")
        except AssertionError as e:
            all_passed = False
            max_diff = np.max(np.abs(t_val - ref_val))
            print(f"[FAIL] {label:<22} Mismatching detected! Max allowed {max_diff:.6e}")

    if all_passed:
        print("\nAll metrics confirmed! PyTorch implementation is 1:1 fit with NumPy.")
    else:
        print("\nThere are misalignment between backend. There are misalignment between backend. Check the tolerance or STFT parameters.")

if __name__ == "__main__":
    run_verification()