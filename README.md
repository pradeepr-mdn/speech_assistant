# Speech Assistant

This is a Raspberry Pi project for **speech-to-text (STT)** and **text-to-speech (TTS)** using:

- **Parakeet** for STT (`nemo-parakeet-tdt-0.6b-v2`)
- **Kokoro TTS** for speech synthesis

---------------------------------------------------------------------------------

## Project Structure

speech_assistant/
├── parakeet_stt.py # STT script
├── main_kokoro_tts.py # TTS script
├── README.md # Project documentation
├── .gitignore # Ignored files/folders
├── parakeet_env/ # Python virtual environment (ignored)
├── models/ # Model files (ignored in Git)
│ ├── kokoro-v1.0.onnx
│ └── voices-v1.0.bin
├── output.wav # Sample output audio (ignored)
└── temp_input.txt # Temporary text input (ignored)

---
---------------------------------------------------------------------------------

## Setup Instructions

1. **Clone the repository:**
git clone https://github.com/pradeepr-mdn/speech_assistant.git
cd speech_assistant

Create a Python virtual environment:
python3 -m venv parakeet_env
source parakeet_env/bin/activate

Install dependencies:
pip install -r requirements.txt
---------------------------------------------------------------------------------
Download the required model and voices files using these commands:
Place the following files in the models/ folder:

version:- kokoro-v1.0.onnx
cmd:- wget https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx


version:- voices-v1.0.bin
cmd :- wget https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin


Note: Models are ignored in Git due to large file size.

---------------------------------------------------------------------------------
Usage
---------------------------------------------------------------------------------

Run STT (Speech-to-Text)
python parakeet_stt.py


The script will start recording audio from your microphone.
By default, it records 3 seconds at a time.
After recording, the audio is saved to a temporary .wav file (temp_input.wav) internally (handled automatically).
The recognized text will be printed in the terminal, e.g.:

Recognized: Hello, this is a test


This allows you to test whether the STT is correctly picking up your voice.
---------------------------------------------------------------------------------
Run TTS (Text-to-Speech)
python main_kokoro_tts.py


The script reads text from temp_input.txt (or the text you provide in the function).
It generates speech output and saves it as output.wav in the project root.
You can play the generated audio using:

aplay output.wav      # on Raspberry Pi


Example: The default text in the script is:

"Welcome to your Raspberry Pi. This is a demonstration of the Kokoro text-to-speech system. Enjoy your voice assistant experience!"
----------------------------------------------------------------------------------


