Dataset

  ↓
  
Audio Sample

  ↓
  
+---------------------+

| Experiment Pipeline |

+---------------------+


1. Moonshine Tiny
2. Whisper Tiny
3. Moonshine + LLM Always-On
4. Whisper + LLM Always-On
5. Moonshine + Confidence Gate + LLM
6. Whisper + Confidence Gate + LLM

# Update package list and install system dependencies
sudo apt update

sudo apt install -y ffmpeg git python3-pip python3-venv build-essential cmake

# Create and activate a Python virtual environment
python3 -m venv venv

source venv/bin/activate

# Install all necessary Python libraries
pip install torch torchaudio

pip install moonshine-voice

pip install openai-whisper

pip install transformers

pip install accelerate

pip install jiwer

pip install psutil

pip install pandas

pip install tqdm

pip install librosa

pip install soundfile

pip install faster-whisper 


# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp 

cd llama.cpp 

cmake -B build 

cmake --build build -j4 

cd ..
