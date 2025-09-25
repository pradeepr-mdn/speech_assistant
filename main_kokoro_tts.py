import kokoro_tts

def synthesize_text_to_speech(text, output_file="output.wav"):
    # Write text to a temporary file first
    with open("temp_input.txt", "w") as f:
        f.write(text)

    # Call Kokoro's function to convert text file to audio file
    kokoro_tts.convert_text_to_audio(
        input_file="temp_input.txt",
        output_file=output_file,
        voice="af_bella",    # choose voice supported in your install
        speed=1.0,
        lang="en-us",
        format="wav",
        model_path="/home/mdn/Desktop/work/speech-assistant/models/kokoro-v1.0.onnx",
        voices_path="/home/mdn/Desktop/work/speech-assistant/models/voices-v1.0.bin"
    )
    print(f"Synthesized audio saved to {output_file}")

if __name__ == "__main__":
    synthesize_text_to_speech("Welcome to your Raspberry Pi. This is a demonstration of the Kokoro text-to-speech system. Enjoy your voice assistant experience!")
