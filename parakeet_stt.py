import sounddevice as sd
import numpy as np
import wave
import tempfile
import onnx_asr

model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2")

def record_audio(duration=3, fs=16000):
    print("Start speaking...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Done recording")
    return audio.flatten()

def save_temp_wav(audio, fs=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        with wave.open(f.name, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(fs)
            wav_file.writeframes(audio.tobytes())
        return f.name

while True:
    audio = record_audio()
    wav_file_path = save_temp_wav(audio)
    text = model.recognize(wav_file_path)
    print("Recognized:", text)
