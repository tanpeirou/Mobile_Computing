import os
from llama_cpp import Llama
os.environ["KERAS_BACKEND"] = "torch"
import torch
import math
from faster_whisper import WhisperModel
import librosa
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForSpeechSeq2Seq

# ==========================================
# 1. Faster-Whisper Tiny (Manual Chunking + Probs)
# ==========================================
whisper_model = WhisperModel("tiny", device="cpu", compute_type="float32")

def whisper_transcribe(audio_path):
    """Manually slices audio into 30s blocks and tracks log-probs."""
    audio, sr = librosa.load(audio_path, sr=16000)
    chunk_samples = 30 * sr
    
    full_text = []
    segment_probs = []
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < sr: continue # Skip empty tails
        
        # Force float32 array for Whisper
        chunk = np.ascontiguousarray(chunk, dtype=np.float32)
        segments, _ = whisper_model.transcribe(chunk, beam_size=1, condition_on_previous_text=False)
        
        for segment in segments:
            full_text.append(segment.text.strip())
            # Convert logprob to standard 0-1 probability percentage
            prob = math.exp(segment.avg_logprob)
            segment_probs.append(prob)
            
    final_text = " ".join(full_text).strip()
    avg_confidence = sum(segment_probs) / len(segment_probs) if segment_probs else 0.0
    
    return final_text, avg_confidence

# ==========================================
# 2. Moonshine Tiny (Manual Chunking + Probs)
# ==========================================
moonshine_processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny")
moonshine_model_pt = AutoModelForSpeechSeq2Seq.from_pretrained("UsefulSensors/moonshine-tiny").to("cpu")

def moonshine_transcribe(audio_path):
    """Manually slices audio into 30s blocks and tracks Hugging Face generation scores."""
    audio, sr = librosa.load(audio_path, sr=16000)
    chunk_samples = 30 * sr
    
    full_text = []
    segment_probs = []
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < sr: continue
        
        inputs = moonshine_processor(chunk, sampling_rate=sr, return_tensors="pt").to("cpu")
        with torch.no_grad():
            # NEW: Request generation scores and dictionary output
            outputs = moonshine_model_pt.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=True
            )
            
        # Decode the generated text
        chunk_text = moonshine_processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        full_text.append(chunk_text.strip())
        
        # NEW: Compute probabilities for the generated tokens
        transition_scores = moonshine_model_pt.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        
        # transition_scores contains log-probs. Convert to linear probabilities (0.0 to 1.0)
        # Take the exponent, then calculate the mean confidence for this chunk
        probs = torch.exp(transition_scores[0]) # [0] because batch size is 1
        avg_chunk_prob = probs.mean().item()
        
        if not math.isnan(avg_chunk_prob):
            segment_probs.append(avg_chunk_prob)
        
    final_text = " ".join(full_text).strip()
    avg_confidence = sum(segment_probs) / len(segment_probs) if segment_probs else 0.0
    
    return final_text, avg_confidence

# ==========================================
# 3. Gemma 2 2B LLM (GGUF via llama.cpp)
# ==========================================
# Point this to the exact path where you downloaded the .gguf file
llm_model_path = "models/gemma-2-2b-it-Q4_K_M.gguf"

print("Loading Gemma 2 2B Model into RAM...")
llm_model = Llama(
    model_path=llm_model_path,
    n_ctx=2048,       # Maximum context window
    n_threads=4,      # Number of CPU cores to use (adjust based on your machine)
    verbose=False     # Set to True if you want to see exactly how fast it processes tokens
)

LLM_CALL_COUNT = 0

def reset_llm_counter():
    global LLM_CALL_COUNT
    LLM_CALL_COUNT = 0

def get_llm_calls():
    global LLM_CALL_COUNT
    return LLM_CALL_COUNT

def process_llm_chunk(text_chunk):
    """Helper function to process a single small block of text."""
    global LLM_CALL_COUNT
    if not text_chunk.strip(): return ""
    
    LLM_CALL_COUNT += 1 
    
    # NEW: Combine the system instruction and the prompt into one string
    prompt = (
        "You are an expert speech text correction assistant. Do not need to add opening and closing double quotation mark.Respond only with the corrected sentence.\n\n"
        "Fix any transcription, grammar, or spelling errors in this text. Maintain the original meaning. "
        f"Only output the corrected text and nothing else:\n\n\"{text_chunk}\""
    )
    
    # NEW: Only pass a "user" role to accommodate Gemma 2
    response = llm_model.create_chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.1 
    )
    
    return response["choices"][0]["message"]["content"].strip()

def llm_correct(corrupted_text):
    """Splits massive transcripts into 50-word blocks so the LLM doesn't truncate."""
    words = corrupted_text.split()
    chunk_size = 50
    corrected_chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        corrected_chunks.append(process_llm_chunk(chunk))
        
    return " ".join(corrected_chunks).strip()
    
# ==========================================
# 4. Confidence Gate Logic
# ==========================================
def confidence_gate(confidence_score, threshold=0.80):
    """
    Returns True if the ASR is confident (skip LLM).
    Returns False if confidence is below threshold (trigger LLM).
    """
    if confidence_score is None:
        return False
    return confidence_score >= threshold
