from kittentts import KittenTTS
import soundfile as sf
import numpy as np

m = KittenTTS("KittenML/kitten-tts-nano-0.2")

texts = [
    "This high quality TTS model works without a GPU.",
    "It is designed to be ultra lightweight and fast.",
    "With expressive voices, the speech sounds natural and clear."
]

print("Starting audio generation for 3 sentences...")

audio_segments = []
for text in texts:
    print(f"Generating audio for: {text}")
    segment = m.generate(text, voice='expr-voice-2-m', speed=1.2)
    audio_segments.append(segment)

full_audio = np.concatenate(audio_segments)

print("Saving concatenated audio to output.wav...")
sf.write('output.wav', full_audio, 24000)

print("Audio saved successfully.")
