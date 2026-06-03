import jiwer
import pandas as pd
from resource_monitor import ResourceMonitor
from sentence_transformers import SentenceTransformer, util

from models import (
    whisper_transcribe, 
    moonshine_transcribe,
    llm_correct, 
    confidence_gate,
    reset_llm_counter, 
    get_llm_calls
)

# ==========================================
# Optimized System Definitions
# ==========================================
def sys1(audio): 
    text, conf = moonshine_transcribe(audio)
    return text

def sys2(audio): 
    text, conf = whisper_transcribe(audio)
    return text

def sys3(audio): 
    text, conf = moonshine_transcribe(audio)
    return llm_correct(text)

def sys4(audio): 
    text, conf = whisper_transcribe(audio)
    return llm_correct(text)

def sys5(audio):
    text, conf = moonshine_transcribe(audio)
    return text if confidence_gate(conf) else llm_correct(text)

def sys6(audio):
    text, conf = whisper_transcribe(audio)
    return text if confidence_gate(conf) else llm_correct(text)

systems = {
    "System 1 (Moonshine)": sys1,
    "System 2 (Whisper)": sys2,
    "System 3 (Moonshine + Always LLM)": sys3,
    "System 4 (Whisper + Always LLM)": sys4,
    "System 5 (Moonshine + Gate + LLM)": sys5,
    "System 6 (Whisper + Gate + LLM)": sys6
}

dataset = [
    ("data/tongue_twister/twister1.wav", "data/groundtruth/twister1.txt"),
    ("data/tongue_twister/twister2.wav", "data/groundtruth/twister2.txt"),
    ("data/tongue_twister/twister3.wav", "data/groundtruth/twister3.txt"),
    ("data/tongue_twister/twister4.wav", "data/groundtruth/twister4.txt"),
    ("data/tongue_twister/twister5.wav", "data/groundtruth/twister5.txt"),
    ("data/tongue_twister/twister6.wav", "data/groundtruth/twister6.txt"),
    ("data/tongue_twister/twister7.wav", "data/groundtruth/twister7.txt"),
    ("data/tongue_twister/twister8.wav", "data/groundtruth/twister8.txt"),
    ("data/tongue_twister/twister9.wav", "data/groundtruth/twister9.txt"),
    ("data/tongue_twister/twister10.wav", "data/groundtruth/twister10.txt"),
    ("data/tongue_twister/twister11.wav", "data/groundtruth/twister11.txt"),
    ("data/tongue_twister/twister12.wav", "data/groundtruth/twister12.txt"),
    ("data/tongue_twister/twister13.wav", "data/groundtruth/twister13.txt"),
    ("data/tongue_twister/twister14.wav", "data/groundtruth/twister14.txt"),
    ("data/tongue_twister/twister15.wav", "data/groundtruth/twister15.txt"),
    ("data/tongue_twister/twister16.wav", "data/groundtruth/twister16.txt"),
    ("data/tongue_twister/twister17.wav", "data/groundtruth/twister17.txt"),
    ("data/tongue_twister/twister18.wav", "data/groundtruth/twister18.txt"),
    ("data/tongue_twister/twister19.wav", "data/groundtruth/twister19.txt"),
    ("data/tongue_twister/twister20.wav", "data/groundtruth/twister20.txt"),
    ("data/tongue_twister/twister21.wav", "data/groundtruth/twister21.txt"),
    ("data/tongue_twister/twister22.wav", "data/groundtruth/twister22.txt"),
    ("data/tongue_twister/twister23.wav", "data/groundtruth/twister23.txt"),
    ("data/tongue_twister/twister24.wav", "data/groundtruth/twister24.txt"),
    ("data/tongue_twister/twister25.wav", "data/groundtruth/twister25.txt"),
    ("data/tongue_twister/twister26.wav", "data/groundtruth/twister26.txt"),
    ("data/tongue_twister/twister27.wav", "data/groundtruth/twister27.txt"),
    ("data/tongue_twister/twister28.wav", "data/groundtruth/twister28.txt"),
    ("data/tongue_twister/twister29.wav", "data/groundtruth/twister29.txt"),
    ("data/tongue_twister/twister30.wav", "data/groundtruth/twister30.txt"),
    ("data/tongue_twister/twister31.wav", "data/groundtruth/twister31.txt"),
    ("data/tongue_twister/twister32.wav", "data/groundtruth/twister32.txt"),
    ("data/tongue_twister/twister33.wav", "data/groundtruth/twister33.txt"),
    ("data/tongue_twister/twister34.wav", "data/groundtruth/twister34.txt"),
    ("data/tongue_twister/twister35.wav", "data/groundtruth/twister35.txt"),
    ("data/tongue_twister/twister36.wav", "data/groundtruth/twister36.txt"),
    ("data/tongue_twister/twister37.wav", "data/groundtruth/twister37.txt"),
    ("data/tongue_twister/twister38.wav", "data/groundtruth/twister38.txt"),
    ("data/tongue_twister/twister39.wav", "data/groundtruth/twister39.txt"),
    ("data/tongue_twister/twister40.wav", "data/groundtruth/twister40.txt"),
    ("data/tongue_twister/twister41.wav", "data/groundtruth/twister41.txt"),
    ("data/tongue_twister/twister42.wav", "data/groundtruth/twister42.txt"),
    ("data/tongue_twister/twister43.wav", "data/groundtruth/twister43.txt"),
    ("data/tongue_twister/twister44.wav", "data/groundtruth/twister44.txt"),
    ("data/tongue_twister/twister45.wav", "data/groundtruth/twister45.txt"),
    ("data/tongue_twister/twister46.wav", "data/groundtruth/twister46.txt"),
    ("data/tongue_twister/twister47.wav", "data/groundtruth/twister47.txt"),
    ("data/tongue_twister/twister48.wav", "data/groundtruth/twister48.txt"),
    ("data/tongue_twister/twister49.wav", "data/groundtruth/twister49.txt"),
    ("data/tongue_twister/twister50.wav", "data/groundtruth/twister50.txt"),
#    ("data/conversation/conversation_1.wav", "data/groundtruth/conversation_1.txt"),
#    ("data/conversation/conversation_2.wav", "data/groundtruth/conversation_2.txt"),
#    ("data/conversation/conversation_3.wav", "data/groundtruth/conversation_3.txt"),
#    ("data/conversation/conversation_4.wav", "data/groundtruth/conversation_4.txt"),
#    ("data/conversation/conversation_5.wav", "data/groundtruth/conversation_5.txt")
]

results = []

# Load the lightweight semantic evaluation model onto the CPU
print("Loading Semantic Similarity Model...")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

for audio_file, gt_file in dataset:
    # Read ground truth file
    with open(gt_file, 'r', encoding='utf-8') as f:
        ground_truth = f.read().strip()
        
    print(f"\n\n{'='*80}")
    print(f"PROCESSING FILE: {audio_file}")
    print(f"{'='*80}")
    
    for name, system in systems.items():
        monitor = ResourceMonitor()
        reset_llm_counter() # Reset the LLM call counter to 0
        monitor.start()
        
        prediction = system(audio_file)
        
        metrics = monitor.stop()
        llm_calls = get_llm_calls()
        
        if not prediction.strip():
            prediction = "[EMPTY OUTPUT]"
            
        # Calculate Strict Error Rates (WER/CER)
        word_error_rate = jiwer.wer(ground_truth, prediction)
        char_error_rate = jiwer.cer(ground_truth, prediction)
        
        # Calculate Semantic Similarity Score (Meaning)
        gt_embedding = semantic_model.encode(ground_truth, convert_to_tensor=True)
        pred_embedding = semantic_model.encode(prediction, convert_to_tensor=True)
        semantic_score = util.cos_sim(gt_embedding, pred_embedding).item()
        
        print(f"\n--- {name} ---")
        print(f"WER: {word_error_rate*100:.1f}% | CER: {char_error_rate*100:.1f}% | Similarity: {semantic_score*100:.1f}%")
        print(f"LLM Calls: {llm_calls} | Latency: {metrics['latency']:.2f}s | Memory Delta: {metrics['memory_mb']:.2f} MB")
        print(f"\nGround Truth: {ground_truth}")
        print(f"System Output: {prediction}")
        
        # Save all metrics AND the texts to our results array
        results.append({
            "system": name,
            "file": audio_file,
            "wer": word_error_rate,
            "cer": char_error_rate,
            "semantic_similarity": semantic_score,
            "llm_calls": llm_calls,
            "latency": metrics["latency"],
            "memory": metrics["memory_mb"],
            "prediction_text": prediction,
            "groundtruth_text": ground_truth  # NEW: Added ground truth directly after prediction
        })

# Save everything to a CSV for your thesis
df = pd.DataFrame(results)
df.to_csv("benchmark_results.csv", index=False)
print("\nExperiment complete! Results saved to benchmark_results.csv")
