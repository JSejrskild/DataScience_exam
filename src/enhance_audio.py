

import os
import torch
import librosa
import soundfile as sf
from speechbrain.inference.enhancement import SpectralMaskEnhancement

# Loading the speechbrain enhancer
enhancer = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus"
)

def enhance_file(file_path, output_path, enhancer):
    signal, sample_rate = librosa.load(file_path, sr=None)
    signal_16k = librosa.resample(signal, orig_sr=sample_rate, target_sr=16000)
    signal_tensor = torch.tensor(signal_16k).unsqueeze(0).float()

    with torch.no_grad():
        enhanced_tensor = enhancer.enhance_batch(signal_tensor, lengths=torch.tensor([1.0]))

    enhanced = enhanced_tensor.squeeze(0).numpy()
    sf.write(output_path, enhanced, 16000)

def enhance_directory(input_dir="../data", output_dir="../enhanced_data"):
    os.makedirs(output_dir, exist_ok=True)
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return

    print(f"Found {len(wav_files)} files — starting enhancement...\n")

    for i, filename in enumerate(wav_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            enhance_file(input_path, output_path, enhancer)
            print(f"[{i}/{len(wav_files)}] ✓ {filename}")
        except Exception as e:
            print(f"[{i}/{len(wav_files)}] ✗ {filename} — ERROR: {e}")

    print(f"\nDone. Enhanced files saved to '{output_dir}'")

enhance_directory(input_dir="../data", output_dir="../enhanced_data")