import jiwer
import pandas as pd
from resource_monitor import ResourceMonitor
from models import (whisper_transcribe, moonshine_transcribe, 
                    llm_correct, confidence_gate,
                    reset_llm_counter, get_llm_calls)

# --- Optimized System Definitions ---
def sys1(audio): return moonshine_transcribe(audio)
def sys2(audio): return whisper_transcribe(audio)
def sys3(audio): return llm_correct(moonshine_transcribe(audio))
def sys4(audio): return llm_correct(whisper_transcribe(audio))
def sys5(audio):
    text = moonshine_transcribe(audio)
    return text if confidence_gate(text) else llm_correct(text)
def sys6(audio):
    text = whisper_transcribe(audio)
    return text if confidence_gate(text) else llm_correct(text)

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
    ("data/conversation/conversation_1.wav", "data/groundtruth/conversation_1.txt"),
    ("data/conversation/conversation_2.wav", "data/groundtruth/conversation_2.txt"),
    ("data/conversation/conversation_3.wav", "data/groundtruth/conversation_3.txt"),
    ("data/conversation/conversation_4.wav", "data/groundtruth/conversation_4.txt"),
    ("data/conversation/conversation_5.wav", "data/groundtruth/conversation_5.txt")
]

results = []

for audio_file, gt_file in dataset:
    ground_truth = open(gt_file).read().strip()
    
    print(f"\n\n{'='*80}")
    print(f"PROCESSING FILE: {audio_file}")
    print(f"{'='*80}")
    
    for name, system in systems.items():
        monitor = ResourceMonitor()
        reset_llm_counter()  # Reset the LLM call counter to 0
        monitor.start()
        
        prediction = system(audio_file)
        metrics = monitor.stop()
        
        llm_calls = get_llm_calls()
        
        if not prediction.strip():
            prediction = "[EMPTY_OUTPUT]"
            
        # Calculate Error Rates
        word_error_rate = jiwer.wer(ground_truth, prediction)
        char_error_rate = jiwer.cer(ground_truth, prediction)
        
        print(f"\n--- {name} ---")
        print(f"WER: {word_error_rate*100:.1f}% | CER: {char_error_rate*100:.1f}% | LLM Calls: {llm_calls}")
        print(f"Latency: {metrics['latency']:.2f}s | Memory Delta: {metrics['memory_mb']:.2f}MB")
        
        # --- NEW: Print the detected text ---
        print(f"\nGround Truth : {ground_truth}")
        print(f"System Output: {prediction}")
        
        # Save all metrics AND the generated text to our results array
        results.append({
            "system": name,
            "file": audio_file,
            "wer": word_error_rate,
            "cer": char_error_rate,
            "llm_calls": llm_calls,
            "latency": metrics["latency"],
            "memory": metrics["memory_mb"],
            "prediction_text": prediction # Saves the text to the CSV
        })

# Save everything to a CSV for your thesis
df = pd.DataFrame(results)
df.to_csv("benchmark_results.csv", index=False)
print("\nExperiment complete! Results saved to benchmark_results.csv")
