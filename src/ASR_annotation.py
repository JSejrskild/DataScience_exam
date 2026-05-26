## This is the pipeline script for transcription of the raw audiofiles
## First, run this in the terminal: sudo apt update && sudo apt install ffmpeg
## Input is a folder with multiple .wav files
## Transcription pipeline for verbal fluency tasks

## VAD (Voice Activity Detection) is to suppress hallucinations, but it is not that good

import os
import csv
from faster_whisper import WhisperModel


def run_whisper_w_prompt(folder_path, model_type="large-v3-turbo", prompt=None, language="da", top_k=1):
    print(f"Loading faster-whisper model '{model_type}'...")
    model = WhisperModel( model_type, device="cpu", compute_type="int8")

    results = {}

    audio_files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    if not audio_files:
        print(f"No files found in {folder_path}")
        return results

    for filename in audio_files:
        file_path = os.path.join(folder_path, filename)
        print(f"\nTranscribing: {filename}")

        try:
            segments, info = model.transcribe(
                file_path,
                language=language,
                initial_prompt=prompt,
                word_timestamps=True,
                beam_size=5,
                
                #VAD filter
                vad_filter=True,                  
                vad_parameters=dict(
                    threshold=0.2,
                    speech_pad_ms=600,
                ),
                no_speech_threshold=0.6         
            )

            # Materialise the generator
            segments = list(segments)

        except Exception as e:
            print(f"  ERROR transcribing {filename}: {e}")
            results[filename] = {"error": str(e)}
            continue

        # Flatten all words across all segments
        word_dicts = []
        all_avg_logprobs = []
        full_text_parts = []

        for seg in segments:
            if seg.avg_logprob is not None:
                all_avg_logprobs.append(seg.avg_logprob)

            full_text_parts.append(seg.text.strip())

            for w in seg.words:
                # faster-whisper doesn't expose per-token alternatives natively,
                # so we store top_k slots; only the chosen word is populated.
                alternatives = [
                    {"token": w.word.strip(), "probability": round(w.probability, 6)}
                ] + [
                    {"token": "", "probability": ""}
                    for _ in range(top_k - 1)
                ]

                word_dicts.append({
                    "word":        w.word.strip(),
                    "start_time":  round(w.start, 2),
                    "probability": round(w.probability, 6),
                    "alternatives": alternatives,
                })

        avg_logprob = float(sum(all_avg_logprobs) / len(all_avg_logprobs)) if all_avg_logprobs else None
        full_text = " ".join(full_text_parts)

        results[filename] = {
            "text":        full_text,
            "language":    language,
            "avg_logprob": avg_logprob,
            "words":       word_dicts,
        }

        n_words = len(word_dicts)
        print(f"  -> {n_words} words, avg_logprob={avg_logprob:.3f}" if avg_logprob else f"  -> {n_words} words")

    return results


def save_results_csv(results, output_path, top_k=1):
    n_alts   = top_k - 1
    alt_cols = []
    for i in range(1, n_alts + 1):
        alt_cols += [f"word_alt_{i}", f"prob_alt_{i}"]

    header = ["filename", "start_time", "word_chosen", "prob_chosen"] + alt_cols

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for filename, data in results.items():
            if "error" in data:
                continue
            for w in data["words"]:
                row = {
                    "filename":    filename,
                    "start_time":  w["start_time"],
                    "word_chosen": w["word"],
                    "prob_chosen": w["probability"],
                }
                for i, alt in enumerate(w["alternatives"][1:n_alts + 1], start=1):
                    row[f"word_alt_{i}"] = alt["token"]
                    row[f"prob_alt_{i}"] = alt["probability"]
                for i in range(len(w["alternatives"]), n_alts + 1):
                    row[f"word_alt_{i}"] = ""
                    row[f"prob_alt_{i}"] = ""

                writer.writerow(row)

    print(f"\nResults saved to {output_path}")

def save_logprob_csv(results, output_path):
    """Save per-file average log probabilities to a separate CSV."""
    header = ["filename", "avg_logprob", "n_words"]
 
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
 
        for filename, data in results.items():
            if "error" in data:
                continue
            writer.writerow({
                "filename":    filename,
                "avg_logprob": round(data["avg_logprob"], 6) if data["avg_logprob"] is not None else "",
                "n_words":     len(data["words"]),
            })
 
    print(f"Log prob summary saved to {output_path}")


results = run_whisper_w_prompt(
    folder_path="../data",
    model_type="large-v3-turbo",
    prompt= None, 
    language="da",
    top_k=1,
)

save_results_csv(results, "../output/ASR_transcription_alldata_enhanced.csv", top_k=1)
save_logprob_csv(results, "../output/ASR_logprob_summary_alldata_enhanced.csv")

