import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
import faster_whisper
import librosa
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForSpeechSeq2Seq

# ==========================================
# 1. Faster-Whisper Tiny (Manual Chunking)
# ==========================================
whisper_model = faster_whisper.WhisperModel("tiny", device="cpu", compute_type="float32")

def whisper_transcribe(audio_path):
    """Manually slices audio into 30s blocks to guarantee 100% processing."""
    audio, sr = librosa.load(audio_path, sr=16000)
    chunk_samples = 30 * sr
    full_text = []
    
    # Loop through audio in 30-second steps
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]
        if len(chunk) < sr: continue # Skip empty tails
        
        # Force float32 array for Whisper
        chunk = np.ascontiguousarray(chunk, dtype=np.float32)
        
        segments, _ = whisper_model.transcribe(chunk, beam_size=1, condition_on_previous_text=False)
        chunk_text = " ".join([segment.text for segment in segments])
        full_text.append(chunk_text.strip())
        
    return " ".join(full_text).strip()

# ==========================================
# 2. Moonshine Tiny (Manual Chunking)
# ==========================================
moonshine_processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny")
moonshine_model_pt = AutoModelForSpeechSeq2Seq.from_pretrained("UsefulSensors/moonshine-tiny").to("cpu")

def moonshine_transcribe(audio_path):
    """Manually slices audio into 30s blocks for Hugging Face."""
    audio, sr = librosa.load(audio_path, sr=16000)
    chunk_samples = 30 * sr
    full_text = []
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]
        if len(chunk) < sr: continue
        
        inputs = moonshine_processor(chunk, sampling_rate=sr, return_tensors="pt").to("cpu")
        with torch.no_grad():
            generated_ids = moonshine_model_pt.generate(**inputs)
            
        chunk_text = moonshine_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        full_text.append(chunk_text.strip())
        
    return " ".join(full_text).strip()

# ==========================================
# 3. Qwen2.5-1.5B LLM (Text Chunking & Call Tracking)
# ==========================================
llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, torch_dtype="auto", device_map="cpu")

# --- NEW: LLM Call Tracker ---
LLM_CALL_COUNT = 0

def reset_llm_counter():
    global LLM_CALL_COUNT
    LLM_CALL_COUNT = 0
    
def get_llm_calls():
    global LLM_CALL_COUNT
    return LLM_CALL_COUNT
# -----------------------------

def process_llm_chunk(text_chunk):
    """Helper function to process a single small block of text."""
    global LLM_CALL_COUNT
    if not text_chunk.strip(): return ""
    
    LLM_CALL_COUNT += 1  # Increment the counter every time the LLM fires!
    
    prompt = f"Fix any transcription, grammar, or spelling errors in this text. Maintain the original meaning. Only output the corrected text and nothing else:\n\n\"{text_chunk}\""
    
    messages = [
        {"role": "system", "content": "You are an expert speech text correction assistant. Respond only with the corrected sentence."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cpu")
    
    generated_ids = llm_model.generate(**model_inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def llm_correct(corrupted_text):
    """Splits massive transcripts into 50-word blocks so the LLM doesn't truncate."""
    words = corrupted_text.split()
    chunk_size = 50
    corrected_chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        corrected_chunks.append(process_llm_chunk(chunk))
        
    return " ".join(corrected_chunks).strip()

# ==========================================
# 4. Confidence Gate Logic
# ==========================================
def confidence_gate(text):
    if len(text.split()) <= 3:
        return False  
    return True
