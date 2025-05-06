import os
import whisper
import librosa
import noisereduce as nr
import soundfile as sf

def denoise\_audio(input\_path, output\_path, noise\_duration=0.5):
"""
Reduce noise in an audio file and save the cleaned version.

```
Args:
    input_path (str): Path to input audio file (WAV).
    output_path (str): Path to save denoised audio.
    noise_duration (float): Duration (in seconds) of noise-only audio at start.

Returns:
    str: Path to the denoised audio file.
"""
print("[INFO] Loading audio for noise reduction...")
y, sr = librosa.load(input_path, sr=None)

print(f"[INFO] Sampling rate: {sr}")
noise_sample = y[:int(noise_duration * sr)]

print("[INFO] Reducing noise...")
reduced_audio = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)

print(f"[INFO] Saving denoised audio to {output_path}...")
sf.write(output_path, reduced_audio, sr)

return output_path
```

def transcribe\_audio(audio\_path, model\_size="base"):
"""
Transcribe speech from an audio file using Whisper.

```
Args:
    audio_path (str): Path to the audio file.
    model_size (str): Whisper model size to use ("tiny", "base", "small", "medium", "large").

Returns:
    str: Transcribed text.
"""
print(f"[INFO] Loading Whisper model ({model_size})...")
model = whisper.load_model(model_size)

print(f"[INFO] Transcribing audio: {audio_path}")
result = model.transcribe(audio_path)

return result['text']
```

def main():
    input\_audio = "input.wav"       # Replace with your audio file
    denoised\_audio = "denoised.wav"

```
# Step 1: Denoise
    denoise_audio(input_audio, denoised_audio)

    # Step 2: Transcribe
    transcription = transcribe_audio(denoised_audio, model_size="base")

    # Step 3: Output
    print("\n=== TRANSCRIPTION RESULT ===\n")
    print(transcription)
    ```

if __name__ == "__main__":
    main()
