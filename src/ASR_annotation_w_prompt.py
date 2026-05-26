import os
import re
import csv
import pandas as pd
from faster_whisper import WhisperModel


DANISH_WORDS_BY_LETTER = {
    "a": ["and", "arm", "avis", "abe", "anker"],
    "b": ["bold", "bog", "bil", "barn", "bord"],
    "d": ["dør", "dag", "due", "drik", "dal"],
    "f": ["fugl", "fisk", "fod", "fri", "flag"],
    "g": ["glas", "gård", "gave", "grøn", "guld"],
    "h": ["hund", "hus", "hånd", "hest", "hat"],
    "k": ["kat", "kage", "ko", "kort", "kniv"],
    "l": ["lys", "løb", "luft", "land", "lam"],
    "m": ["mus", "mad", "mål", "mand", "mark"],
    "n": ["næse", "nøgle", "nat", "nord", "nål"],
    "p": ["pil", "pest", "pige", "plov", "pude"],
    "r": ["rose", "råb", "regn", "ring", "rov"],
    "s": ["sol", "skov", "sølv", "seng", "slange"],
    "t": ["tog", "træ", "tand", "telt", "tid"],
    "v": ["vand", "vej", "vind", "vogn", "væg"],
}


def get_prompt_for_letter(letter):
    if pd.isna(letter):
        return None
    words = DANISH_WORDS_BY_LETTER.get(letter.strip().lower())
    if not words:
        return None
    return ". ".join(words) + "."



def extract_vf_key(filename):
    if pd.isna(filename):
        return None
    match = re.search(r"(VF-\d{3}-\d{14})", str(filename))
    return match.group(1) if match else None


def load_annotation_lookup(annotations_csv, encoding="latin-1"):
    ann_df = pd.read_csv(annotations_csv, encoding=encoding, sep=";")
    ann_df["_key"] = ann_df["Filename"].apply(extract_vf_key)
    ann_df = ann_df[ann_df["_key"].notna()]
    letter_lookup = dict(zip(ann_df["_key"], ann_df["Target Letter"]))
    annotated_keys = set(ann_df["_key"])
    print(f"Loaded {len(annotated_keys)} annotated files from {annotations_csv}")
    return annotated_keys, letter_lookup


def transcribe_files(folder_path, annotated_keys, letter_lookup, model, language="da", top_k=1):
    audio_files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    results = {}
    skipped = 0

    for filename in audio_files:
        key = extract_vf_key(filename)

        if not key:
            print(f"  SKIP (no VF key parsed): {filename}")
            skipped += 1
            continue

        if key not in annotated_keys:
            print(f"  SKIP (not in annotation CSV): {filename}")
            skipped += 1
            continue

        letter = letter_lookup.get(key)
        prompt = get_prompt_for_letter(letter) if letter else None
        print(f"\nTranscribing: {filename} | key={key} | letter={letter} | prompt={prompt}")

        try:
            segments, _ = model.transcribe(
                os.path.join(folder_path, filename),
                language=language,
                initial_prompt=prompt,
                word_timestamps=True,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.2,
                    speech_pad_ms=600,
                ),
                no_speech_threshold=0.6,
            )
            segments = list(segments)

        except Exception as e:
            print(f"  ERROR transcribing {filename}: {e}")
            results[filename] = {"error": str(e)}
            continue

        word_dicts = []
        all_avg_logprobs = []
        full_text_parts = []

        for seg in segments:
            if seg.avg_logprob is not None:
                all_avg_logprobs.append(seg.avg_logprob)
            full_text_parts.append(seg.text.strip())
            for w in seg.words:
                alternatives = [
                    {"token": w.word.strip(), "probability": round(w.probability, 6)}
                ] + [{"token": "", "probability": ""} for _ in range(top_k - 1)]

                word_dicts.append({
                    "word":         w.word.strip(),
                    "start_time":   round(w.start, 2),
                    "probability":  round(w.probability, 6),
                    "alternatives": alternatives,
                })

        avg_logprob = float(sum(all_avg_logprobs) / len(all_avg_logprobs)) if all_avg_logprobs else None

        results[filename] = {
            "text":        " ".join(full_text_parts),
            "language":    language,
            "avg_logprob": avg_logprob,
            "words":       word_dicts,
            "letter":      letter,
            "key":         key,
        }

        n_words = len(word_dicts)
        print(f"  -> {n_words} words, avg_logprob={avg_logprob:.3f}" if avg_logprob else f"  -> {n_words} words")

    print(f"\nDone. Transcribed: {len(results)} | Skipped: {skipped}")
    return results


def save_results_csv(results, output_path, top_k=1):
    n_alts   = top_k - 1
    alt_cols = []
    for i in range(1, n_alts + 1):
        alt_cols += [f"word_alt_{i}", f"prob_alt_{i}"]

    header = ["filename", "key", "letter", "start_time", "word_chosen", "prob_chosen"] + alt_cols

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for filename, data in results.items():
            if "error" in data:
                continue
            for w in data["words"]:
                row = {
                    "filename":    filename,
                    "key":         data.get("key", ""),
                    "letter":      data.get("letter", ""),
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

    print(f"Results saved to {output_path}")


def save_logprob_csv(results, output_path):
    header = ["filename", "key", "letter", "avg_logprob", "n_words"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for filename, data in results.items():
            if "error" in data:
                continue
            writer.writerow({
                "filename":    filename,
                "key":         data.get("key", ""),
                "letter":      data.get("letter", ""),
                "avg_logprob": round(data["avg_logprob"], 6) if data["avg_logprob"] is not None else "",
                "n_words":     len(data["words"]),
            })

    print(f"Log prob summary saved to {output_path}")


os.makedirs("../output", exist_ok=True)

annotated_keys, letter_lookup = load_annotation_lookup("../data/annotation(Merged).csv")


model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
results = transcribe_files(folder_path="../enhanced_data", annotated_keys=annotated_keys, letter_lookup=letter_lookup, model=model,language="da",top_k=1,)

save_results_csv(results, "../output/ASR_transcription_alldata_enhanced_wprompt.csv", top_k=1)
save_logprob_csv(results, "../output/ASR_logprob_summary_alldata_enhanced_wprompt.csv")

