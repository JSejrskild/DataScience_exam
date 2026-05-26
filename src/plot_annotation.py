# Here is a script for plotting the ASR annotation
# You run it in the bottom (chose what to run)import os
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import jellyfish
import spacy
 
nlp = spacy.load("da_core_news_md")
DANISH_VOWELS = set("aeiouyæøå")

 
def load_data(ann_info, asr_info):
    data_annotated = pd.read_csv(
        os.path.join(ann_info["dir"], ann_info["file"]), sep=";", encoding="latin-1"
    )
    data_asr = pd.read_csv(
        os.path.join(asr_info["dir"], asr_info["file"]), sep=",", encoding="latin-1"
    )
    return data_annotated, data_asr
 
 
def clean_dataframes(data_annotated, data_asr, ann_info, asr_info):
    ann = data_annotated.copy()
    for col in ann.select_dtypes(include="str").columns:
        ann[col] = ann[col].fillna("").str.lower().str.strip()
    ann[ann_info["time"]] = pd.to_numeric(ann[ann_info["time"]], errors="coerce")
    ann[ann_info["filename"]] = ann[ann_info["filename"]].str.replace(r"^-+", "", regex=True)
    ann[ann_info["filename"]] = ann[ann_info["filename"]].str.extract(r"(\d{14}\.wav)")[0]
    ann = ann[ann[ann_info["filename"]].notna() & (ann[ann_info["filename"]] != "")]
 
    asr = data_asr.copy()
    for col in asr.select_dtypes(include="str").columns:
        asr[col] = asr[col].str.lower().str.strip()
    asr[asr_info["word"]] = asr[asr_info["word"]].str.replace(r"[^a-z0-9æøå'\-]", "", regex=True)
    asr[asr_info["time"]] = pd.to_numeric(asr[asr_info["time"]], errors="coerce")
    asr = asr[asr[asr_info["time"]] <= 60].copy()
    asr[asr_info["filename"]] = asr[asr_info["filename"]].str.replace(r"^\d+_[^_]+_", "", regex=True)
 
    # Hallucination filter: remove exact duplicates within 500ms or bursts within 50ms
    asr = asr.sort_values([asr_info["filename"], asr_info["time"]]).reset_index(drop=True)
    prev_word = asr[asr_info["word"]].shift(1)
    prev_time = asr[asr_info["time"]].shift(1)
    prev_file = asr[asr_info["filename"]].shift(1)
    same_file     = asr[asr_info["filename"]] == prev_file
    burst         = same_file & ((asr[asr_info["time"]] - prev_time).abs() < 0.05)
    duplicate     = same_file & (asr[asr_info["word"]] == prev_word) & ((asr[asr_info["time"]] - prev_time).abs() < 0.5)
    n_removed = (burst | duplicate).sum()
    if n_removed:
        print(f"Hallucination filter: removed {n_removed} word(s)")
    asr = asr[~(burst | duplicate)].reset_index(drop=True)
 
    # Normalise ASR to bare timestamp key for matching
    asr[asr_info["filename"]] = asr[asr_info["filename"]].str.extract(r"(\d{14}\.wav)")[0]
 
    return ann, asr
 
 
def plot_timing(data_annotated_clean, data_asr_clean, ann_info, asr_info, dir_info):
    filenames = sorted(
        set(data_annotated_clean[ann_info["filename"]]) &
        set(data_asr_clean[asr_info["filename"]])
    )
    skipped = sorted(
        set(data_asr_clean[asr_info["filename"]]) -
        set(data_annotated_clean[ann_info["filename"]])
    )
    if skipped:
        print(f"plot_timing: skipping {len(skipped)} unannotated ASR file(s): {skipped}")
 
    for fname in filenames:
        ann = (data_annotated_clean[data_annotated_clean[ann_info["filename"]] == fname]
               .sort_values(ann_info["time"]).reset_index(drop=True))
        asr = (data_asr_clean[data_asr_clean[asr_info["filename"]] == fname]
               .sort_values(asr_info["time"]).reset_index(drop=True))
        subject_id = ann[ann_info["id"]].iloc[0] if ann_info.get("id") and ann_info["id"] in ann.columns and len(ann) else fname
 
        n_rows = max(len(ann), len(asr), 1)
        fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * n_rows)))
 
        for i, row in ann.iterrows():
            ax.vlines(row[ann_info["time"]], i - 0.4, i + 0.4, color="#1f77b4", linewidth=2)
            ax.text(row[ann_info["time"]], i + 0.45, str(row[ann_info["word"]]),
                    fontsize=7, color="#1f77b4", ha="center", va="bottom", clip_on=True)
 
        for i, row in asr.iterrows():
            ax.vlines(row[asr_info["time"]], i - 0.4, i + 0.4, color="#ff7f0e", linewidth=2, linestyle="--")
            ax.text(row[asr_info["time"]], i - 0.45, str(row[asr_info["word"]]),
                    fontsize=7, color="#ff7f0e", ha="center", va="top", clip_on=True)
 
        ax.legend(handles=[
            mpatches.Patch(color="#1f77b4", label="Manual annotation"),
            mpatches.Patch(color="#ff7f0e", label="Whisper ASR"),
        ], loc="upper right")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Word index")
        ax.set_title(f"Word onset timings — {subject_id} ({fname})")
        ax.set_ylim(-1, n_rows + 1)
        plt.tight_layout()
        safe = fname.replace("/", "_").replace("\\", "_")
        subj_dir = os.path.join(dir_info["figures"], subject_id)
        os.makedirs(subj_dir, exist_ok=True)
        plt.savefig(os.path.join(subj_dir, f"timing_{safe}.png"), dpi=150)
        print(f"Saved: {subject_id}/timing_{safe}.png")
        plt.show()
 
 
def make_align_events(ann_to_asr, ann_words, ann_times, asr_words, asr_times, asr_probs):
    asr_used = set(ann_to_asr.values())
    events = []
    for ai in range(len(ann_words)):
        if ai in ann_to_asr:
            ji = ann_to_asr[ai]
            events.append((ann_times[ai] if ann_times[ai] is not None else float("inf"), "match", ai, ji))
        else:
            events.append((ann_times[ai] if ann_times[ai] is not None else float("inf"), "ann_only", ai, None))
    for ji in range(len(asr_words)):
        if ji not in asr_used:
            events.append((asr_times[ji] if asr_times[ji] is not None else float("inf"), "asr_only", None, ji))
    events.sort(key=lambda e: e[0])
    pairs = []
    for _, kind, ai, ji in events:
        if kind == "match":
            pairs.append(((ann_words[ai], ann_times[ai]), (asr_words[ji], asr_times[ji], asr_probs[ji])))
        elif kind == "ann_only":
            pairs.append(((ann_words[ai], ann_times[ai]), (None, None, None)))
        elif kind == "asr_only":
            pairs.append(((None, None), (asr_words[ji], asr_times[ji], asr_probs[ji])))
    return pairs
 
 
def align_string(ann_words, ann_times, asr_words, asr_times, asr_probs):
    ann_str = [w if isinstance(w, str) else "⟨INAUDIBLE⟩" for w in ann_words]
    asr_str = [w if isinstance(w, str) else "" for w in asr_words]
    ann_to_asr = {}
    used_asr = set()
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, ann_str, asr_str, autojunk=False).get_opcodes():
        if tag in ("equal", "replace"):
            for i, j in zip(range(i1, i2), range(j1, j2)):
                ann_to_asr[i] = j
                used_asr.add(j)
    return make_align_events(ann_to_asr, ann_words, ann_times, asr_words, asr_times, asr_probs)
 
 
def align_greedy(ann_words, ann_times, asr_words, asr_times, asr_probs, time_window=5.0):
    used_asr = set()
    ann_to_asr = {}
    for ai, (aw, at) in enumerate(zip(ann_words, ann_times)):
        if at is None:
            continue
        best_ji, best_td, best_sim = None, float("inf"), -1
        for ji, (bw, bt) in enumerate(zip(asr_words, asr_times)):
            if ji in used_asr or bt is None or not isinstance(bw, str):
                continue
            td = abs(at - bt)
            if td > time_window:
                continue
            sim = SequenceMatcher(None, aw, bw).ratio() if (isinstance(aw, str) and isinstance(bw, str)) else 0.0
            if td < best_td or (td == best_td and sim > best_sim):
                best_ji, best_td, best_sim = ji, td, sim
        if best_ji is not None:
            ann_to_asr[ai] = best_ji
            used_asr.add(best_ji)
    return make_align_events(ann_to_asr, ann_words, ann_times, asr_words, asr_times, asr_probs)
 
 
def align_hungarian(ann_words, ann_times, asr_words, asr_times, asr_probs, time_window=5.0):
    from scipy.optimize import linear_sum_assignment
    n, m = len(ann_words), len(asr_words)
    BIG = 1e9
    cost = np.full((n, m), BIG)
    for ai, at in enumerate(ann_times):
        if at is None:
            continue
        for ji, bt in enumerate(asr_times):
            if bt is None:
                continue
            td = abs(at - bt)
            if td <= time_window:
                cost[ai, ji] = td
    row_ind, col_ind = linear_sum_assignment(cost)
    ann_to_asr = {}
    for ai, ji in zip(row_ind, col_ind):
        if cost[ai, ji] < BIG:
            ann_to_asr[ai] = ji
    return make_align_events(ann_to_asr, ann_words, ann_times, asr_words, asr_times, asr_probs)
 
 
def compute_pairs(data_annotated_clean, data_asr_clean, ann_info, asr_info, method="greedy"):
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()
 
    def phonetic_sim(a, b):
        return SequenceMatcher(None, jellyfish.metaphone(a), jellyfish.metaphone(b)).ratio()
 
    def semantic_sim(a, b):
        tok_a, tok_b = nlp(a), nlp(b)
        if tok_a.has_vector and tok_b.has_vector:
            return float(np.clip(tok_a.similarity(tok_b), 0, 1))
        return None
 
    filenames = sorted(
        set(data_annotated_clean[ann_info["filename"]]) &
        set(data_asr_clean[asr_info["filename"]])
    )
    skipped = sorted(
        set(data_asr_clean[asr_info["filename"]]) -
        set(data_annotated_clean[ann_info["filename"]])
    )
    if skipped:
        print(f"compute_pairs: skipping {len(skipped)} unannotated ASR file(s): {skipped}")
 
    all_records = []
 
    for fname in filenames:
        ann_df = data_annotated_clean[data_annotated_clean[ann_info["filename"]] == fname].sort_values(ann_info["time"])
        asr_df = data_asr_clean[data_asr_clean[asr_info["filename"]] == fname].sort_values(asr_info["time"])
 
        if len(ann_df) == 0:
            print(f"  compute_pairs: no annotation rows for {fname}, skipping")
            continue
        if len(asr_df) == 0:
            print(f"  compute_pairs: no ASR rows for {fname}, skipping")
            continue
 
        ann_words       = ann_df[ann_info["word"]].astype(str).tolist()
        ann_times       = ann_df[ann_info["time"]].tolist()
        target_letters  = ann_df["Target Letter"].astype(str).str.lower().str.strip().tolist() if "Target Letter" in ann_df.columns else [None] * len(ann_words)
        audio_qualities = ann_df["Audio Quality Comment"].astype(str).str.lower().str.strip().tolist() if "Audio Quality Comment" in ann_df.columns else [None] * len(ann_words)
        certainties     = ann_df["Certainty"].tolist() if "Certainty" in ann_df.columns else [None] * len(ann_words)
 
        asr_valid = asr_df[asr_df[asr_info["word"]].apply(lambda x: isinstance(x, str) and x.strip() != "")].copy()
        asr_words = asr_valid[asr_info["word"]].tolist()
        asr_times = asr_valid[asr_info["time"]].tolist()
        asr_probs = asr_valid[asr_info["prob"]].tolist() if asr_info.get("prob") and asr_info["prob"] in asr_valid.columns else [None] * len(asr_words)
 
        subject_id = ann_df[ann_info["id"]].iloc[0] if ann_info.get("id") and ann_info["id"] in ann_df.columns else fname
 
        ann_words_clean = [None if (not isinstance(w, str) or w.strip() in ("nan", "none", "")) else w for w in ann_words]
 
        if method == "string":
            aligned = align_string(ann_words_clean, ann_times, asr_words, asr_times, asr_probs)
        elif method == "hungarian":
            aligned = align_hungarian(ann_words_clean, ann_times, asr_words, asr_times, asr_probs)
        else:
            aligned = align_greedy(ann_words_clean, ann_times, asr_words, asr_times, asr_probs)
        ann_idx_counter = 0
        for idx, ((a, a_time), (b, b_time, prob)) in enumerate(aligned):
            if a is not None or a_time is not None:
                ann_pos = ann_idx_counter if ann_idx_counter < len(target_letters) else None
                ann_idx_counter += 1
            else:
                ann_pos = None
 
            target_letter = target_letters[ann_pos] if ann_pos is not None else None
            audio_quality = audio_qualities[ann_pos] if ann_pos is not None else None
            certainty     = certainties[ann_pos] if ann_pos is not None else None
 
            inaudible = (a is None and a_time is not None)
 
            char_s = similarity(a, b) if isinstance(a, str) and isinstance(b, str) else None
            phon_s = phonetic_sim(a, b) if isinstance(a, str) and isinstance(b, str) else None
            sem_s  = semantic_sim(a, b) if isinstance(a, str) and isinstance(b, str) else None
 
            if inaudible:
                exact = None
            elif a is not None and b is not None:
                exact = 1 if a == b else 0
            elif a is not None and b is None:
                exact = 0
            else:
                exact = None
 
            time_diff = round(abs(a_time - b_time), 2) if a_time is not None and b_time is not None else None
 
            is_vowel = None
            if target_letter and isinstance(target_letter, str) and target_letter not in ("nan", "none", ""):
                is_vowel = target_letter[0] in DANISH_VOWELS
 
            all_records.append({
                "filename":      fname,
                "subject_id":    subject_id,
                "annotated":     a if a is not None else ("⟨INAUDIBLE⟩" if inaudible else "⟨MISSING⟩"),
                "asr":           b if b is not None else "⟨MISSING⟩",
                "ann_time":      a_time,
                "asr_time":      b_time,
                "time_diff":     time_diff,
                "exact":         exact,
                "char_sim":      char_s,
                "phonetic_sim":  phon_s,
                "semantic_sim":  sem_s,
                "prob_chosen":   prob,
                "target_letter": target_letter,
                "is_vowel":      is_vowel,
                "audio_quality": audio_quality,
                "certainty":     certainty,
                "method":        method,
            })
 
    df = pd.DataFrame(all_records)
    df["annotated"] = df["annotated"].replace({"nan": "⟨INAUDIBLE⟩", "none": "⟨INAUDIBLE⟩"})
    df["asr"]       = df["asr"].replace({"nan": "⟨MISSING⟩", "none": "⟨MISSING⟩"})
    return df
 
 
def plot_alignment_comparison(data_annotated_clean, data_asr_clean, ann_info, asr_info, dir_info):
    colours = {"string": "#1f77b4", "greedy": "#ff7f0e", "hungarian": "#2ca02c"}
    labels  = {"string": "String (original)", "greedy": "Greedy time", "hungarian": "Hungarian time"}
 
    all_dfs = {}
    for method in ["string", "greedy", "hungarian"]:
        print(f"Computing pairs: {method}...")
        df = compute_pairs(data_annotated_clean, data_asr_clean, ann_info, asr_info, method=method)
        all_dfs[method] = df
 
    subjects = sorted(set().union(*[set(df["subject_id"]) for df in all_dfs.values()]))
    x = np.arange(len(subjects))
    width = 0.25
 
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(subjects)), 6))
 
    for i, method in enumerate(["string", "greedy", "hungarian"]):
        df = all_dfs[method]
        df_valid = df[df["exact"].notna() & df["annotated"].ne("⟨INAUDIBLE⟩")]
        per_subj = df_valid.groupby("subject_id")["exact"].mean().reindex(subjects).fillna(0)
        grand_mean = per_subj.mean()
        bars = ax.bar(x + i * width, per_subj.values, width, label=f"{labels[method]} (mean={grand_mean:.2f})",
                      color=colours[method], alpha=0.8, edgecolor="white")
        ax.axhline(grand_mean, color=colours[method], linewidth=1.2, linestyle="--", alpha=0.7)
 
    ax.set_xticks(x + width)
    ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean exact accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Alignment method comparison — participant accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_info["figures"], "alignment_comparison.png"), dpi=150, bbox_inches="tight")
    print("Saved: alignment_comparison.png")
    plt.show()
 
    return all_dfs
 
 
def plot_accuracy(pairs_df, dir_info):
    for fname, group in pairs_df.groupby("filename"):
        subject_id = group["subject_id"].iloc[0]
        ann_n = group["annotated"].ne("⟨MISSING⟩").sum()
        asr_n = group["asr"].ne("⟨MISSING⟩").sum()
 
        def fmt(v, is_exact=False):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            if is_exact:
                return str(int(v))
            return f"{v:.2f}"
 
        display = group[["annotated", "asr", "exact", "time_diff", "char_sim", "phonetic_sim", "semantic_sim"]].copy()
        display.columns = ["Annotated", "ASR", "Exact", "Time diff", "Char sim", "Phonetic", "Semantic"]
        display["Exact"]     = display["Exact"].apply(lambda v: fmt(v, is_exact=True))
        display["Time diff"] = display["Time diff"].apply(fmt)
        display["Char sim"]  = display["Char sim"].apply(fmt)
        display["Phonetic"]  = display["Phonetic"].apply(fmt)
        display["Semantic"]  = display["Semantic"].apply(fmt)
 
        exact_col = list(display.columns).index("Exact")
        time_col  = list(display.columns).index("Time diff")
        char_col  = list(display.columns).index("Char sim")
        phon_col  = list(display.columns).index("Phonetic")
        sem_col   = list(display.columns).index("Semantic")
 
        cell_colours = []
        for _, row in display.iterrows():
            colours = [(1.0, 1.0, 1.0)] * len(display.columns)
 
            if row["Exact"] == "1":
                colours[exact_col] = (0.7, 1.0, 0.7)
            elif row["Exact"] == "0":
                colours[exact_col] = (1.0, 0.7, 0.7)
 
            # Time diff: green for small, red for large (scale 0-5s)
            try:
                t = float(row["Time diff"])
                norm = min(t / 5.0, 1.0)
                colours[time_col] = (0.5 + 0.5 * norm, 1.0 - 0.3 * norm, 0.5)
            except (ValueError, TypeError):
                pass
 
            for col_idx, col_name in [(char_col, "Char sim"), (phon_col, "Phonetic"), (sem_col, "Semantic")]:
                try:
                    v = float(row[col_name])
                    colours[col_idx] = (0.5 + 0.5 * (1 - v), 0.5 + 0.5 * v, 0.5)
                except (ValueError, TypeError):
                    pass
 
            cell_colours.append(colours)
 
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * (len(display) + 2))))
        ax.axis("off")
        tbl = ax.table(
            cellText=display.values.tolist(),
            colLabels=list(display.columns),
            cellColours=cell_colours,
            loc="center",
            cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.auto_set_column_width(col=list(range(len(display.columns))))
        ax.set_title(
            f"Word accuracy — {subject_id} ({fname})  "
            f"(annotation n={ann_n}, ASR n={asr_n})",
            pad=14,
        )
        plt.tight_layout()
        safe = fname.replace("/", "_").replace("\\", "_")
        subj_dir = os.path.join(dir_info["figures"], subject_id)
        os.makedirs(subj_dir, exist_ok=True)
        plt.savefig(os.path.join(subj_dir, f"accuracy_{safe}.png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {subject_id}/accuracy_{safe}.png")
        plt.show()
 
 
def plot_prob_correlations(pairs_df, dir_info):
    df = pairs_df.dropna(subset=["prob_chosen", "char_sim", "phonetic_sim", "semantic_sim"]).copy()
 
    metrics = [
        ("char_sim",     "Char Sim"),
        ("phonetic_sim", "Phonetic Sim"),
        ("semantic_sim", "Semantic Sim"),
    ]
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ASR prob_chosen vs similarity metrics (all words)", fontsize=13)
 
    for ax, (col, label) in zip(axes, metrics):
        x = df[col].astype(float)
        y = df["prob_chosen"].astype(float)
        ax.scatter(x, y, alpha=0.4, s=20, color="#1f77b4")
        m, b = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, m * xline + b, color="#ff7f0e", linewidth=1.5)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(label)
        ax.set_ylabel("prob_chosen")
        ax.set_title(f"r = {r:.3f}")
 
    plt.tight_layout()
    plt.savefig(os.path.join(dir_info["figures"], "prob_correlations.png"), dpi=150)
    print("Saved: prob_correlations.png")
    plt.show()
 
 
def plot_exact_boxplot(pairs_df, dir_info):
    df = pairs_df.dropna(subset=["exact", "prob_chosen"]).copy()
    df["exact"] = df["exact"].astype(int)
 
    fig, ax = plt.subplots(figsize=(7, 5))
 
    groups = [df[df["exact"] == v]["prob_chosen"].astype(float).values for v in [0, 1]]
    labels = ["Incorrect", "Correct"]
 
    bp = ax.boxplot(groups, positions=[0, 1], widths=0.4, patch_artist=True,
                    boxprops=dict(facecolor="#d0e8ff", color="#333"),
                    medianprops=dict(color="#ff7f0e", linewidth=2),
                    whiskerprops=dict(color="#333"),
                    capprops=dict(color="#333"),
                    flierprops=dict(marker=""))
 
    for i, (vals, pos) in enumerate(zip(groups, [0, 1])):
        jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(pos + jitter, vals, alpha=0.4, s=18, color="#1f77b4", zorder=2)
 
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Whisper's internal probabilities (Certainty)")
    ax.set_title("ASR confidence by exact match")
    plt.tight_layout()
    plt.savefig(os.path.join(dir_info["figures"], "boxplot_exact.png"), dpi=150)
    print("Saved: boxplot_exact.png")
    plt.show()
 
 
def plot_exact_boxplot_by_letter_type(pairs_df, dir_info):
    df = pairs_df.dropna(subset=["exact", "prob_chosen", "is_vowel"]).copy()
    df["exact"] = df["exact"].astype(int)
 
    vowel_color    = "#e377c2"
    consonant_color = "#2ca02c"
 
    fig, ax = plt.subplots(figsize=(7, 5))
 
    positions = [0, 1]
    groups = [df[df["exact"] == v]["prob_chosen"].astype(float) for v in [0, 1]]
 
    bp = ax.boxplot(
        [g.values for g in groups],
        positions=positions,
        widths=0.4,
        patch_artist=True,
        boxprops=dict(facecolor="#eeeeee", color="#333"),
        medianprops=dict(color="#ff7f0e", linewidth=2),
        whiskerprops=dict(color="#333"),
        capprops=dict(color="#333"),
        flierprops=dict(marker=""),
        zorder=1,
    )
 
    for pos, exact_val in zip(positions, [0, 1]):
        subset = df[df["exact"] == exact_val]
        for _, row in subset.iterrows():
            color = vowel_color if row["is_vowel"] else consonant_color
            jitter = np.random.uniform(-0.12, 0.12)
            ax.scatter(pos + jitter, row["prob_chosen"], color=color, alpha=0.5, s=20, zorder=2)
 
    ax.set_xticks(positions)
    ax.set_xticklabels(["Not exact (0)", "Exact (1)"])
    ax.set_ylabel("prob_chosen (ASR)")
    ax.set_title("ASR confidence by exact match — coloured by target letter type")
    ax.legend(handles=[
        mpatches.Patch(color=vowel_color,    label="Vowel"),
        mpatches.Patch(color=consonant_color, label="Consonant"),
    ], loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(dir_info["figures"], "boxplot_exact_letter_type.png"), dpi=150)
    print("Saved: boxplot_exact_letter_type.png")
    plt.show()
 
 
def plot_confusion_matrices(pairs_df, dir_info):
    df = pairs_df.dropna(subset=["exact"]).copy()
    df["exact"] = df["exact"].astype(int)
 
    quality_order   = ["low", "medium", "high"]
    certainty_order = [1, 2, 3]
    exact_order     = [0, 1]
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Correctly annotated words by ertainty or audio quality", fontsize=13)
 
    def draw_matrix(ax, df, col, row_labels, title, col_label):
        matrix = pd.crosstab(
            df[col].astype(str).str.lower().str.strip() if df[col].dtype == object else df[col],
            df["exact"],
            rownames=[col],
            colnames=["exact"],
        ).reindex(index=[str(r) if df[col].dtype == object else r for r in row_labels], columns=exact_order, fill_value=0)
 
        vals = matrix.values
        im = ax.imshow(vals, cmap="Blues", aspect="auto")
 
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Incorrrectly annotated", "Corrrectly annotated"])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Correct")
        ax.set_ylabel(col_label)
        ax.set_title(title)
 
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, str(vals[i, j]), ha="center", va="center",
                        color="white" if vals[i, j] > vals.max() * 0.6 else "black", fontsize=11)
 
    draw_matrix(
        axes[0], df, "audio_quality",
        row_labels=quality_order,
        title="Correctly annotated by Audio Quality",
        col_label="Audio Quality Comment",
    )
    draw_matrix(
        axes[1], df, "certainty",
        row_labels=certainty_order,
        title="Correctly annotated by Certainty",
        col_label="Certainty",
    )
 
    plt.tight_layout()
    plt.savefig(os.path.join(dir_info["figures"], "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    print("Saved: confusion_matrices.png")
    plt.show()
 
 
def plot_participant_accuracy(pairs_df, dir_info):
    df = pairs_df[pairs_df["exact"].notna() & pairs_df["annotated"].ne("⟨INAUDIBLE⟩")].copy()
    df["exact"] = df["exact"].astype(float)

    per_subject = df.groupby("subject_id").agg(
        mean_accuracy=("exact", "mean"),
        audio_quality=("audio_quality", lambda x: x.mode()[0] if x.notna().any() else "unknown")
    ).reset_index()
    per_subject = per_subject.sort_values("mean_accuracy", ascending=False)

    grand_mean = per_subject["mean_accuracy"].mean()

    quality_colours = {
            "low":     "#d62728",
            "medium":  "#FFD700",
            "high":    "#2ca02c",
            "unknown": "#aec7e8",
    }

    bar_colours = [
        quality_colours.get(q, "#aec7e8")
        for q in per_subject["audio_quality"]
    ]

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(per_subject)), 5))

    ax.bar(
        range(len(per_subject)),
        per_subject["mean_accuracy"],
        color=bar_colours,
        alpha=0.85,
        edgecolor="white",
    )

    ax.axhline(grand_mean, color="black", linewidth=1.5, linestyle="--",
               label=f"Grand mean: {grand_mean:.2f}")

    legend_handles = [
        mpatches.Patch(color=quality_colours["low"],     label="Low quality"),
        mpatches.Patch(color=quality_colours["medium"],  label="Medium quality"),
        mpatches.Patch(color=quality_colours["high"],    label="High quality"),
        mpatches.Patch(color=quality_colours["unknown"], label="Unknown"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                   label=f"Grand mean: {grand_mean:.2f}"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    ax.set_xticks(range(len(per_subject)))
    ax.set_xticklabels(per_subject["subject_id"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean exact accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Mean ASR accuracy per participant (coloured by audio quality)")
    plt.tight_layout()
    plt.savefig(os.path.join(dir_info["figures"], "participant_accuracy.png"), dpi=150, bbox_inches="tight")
    print("Saved: participant_accuracy.png")
    plt.show()

def plot_wer_comparison(data_annotated_clean, ann_info, dir_info,
                        asr_configs, method="string"):
    from scipy import stats

    colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    all_pairs = {}
    for cfg in asr_configs:
        asr_df = pd.read_csv(
            os.path.join(cfg["dir"], cfg["file"]), sep=",", encoding="latin-1"
        )
        asr_info_local = {
            "filename": "filename",
            "word":     "word_chosen",
            "time":     "start_time",
            "prob":     "prob_chosen",
        }
        _, asr_clean = clean_dataframes(data_annotated, asr_df, ann_info, asr_info_local)
        pairs = compute_pairs(data_annotated_clean, asr_clean, ann_info, asr_info_local, method=method)
        all_pairs[cfg["label"]] = pairs

    labels   = [cfg["label"] for cfg in asr_configs]
    subjects = sorted(set().union(*[set(df["subject_id"]) for df in all_pairs.values()]))
    n_conds  = len(labels)
    n_subj   = len(subjects)

    per_subj = {}
    grand    = {}
    for label in labels:
        df = all_pairs[label]
        df_valid = df[df["exact"].notna() & df["annotated"].ne("⟨INAUDIBLE⟩")].copy()
        df_valid["exact"] = df_valid["exact"].astype(float)
        per_subj[label] = df_valid.groupby("subject_id")["exact"].mean().reindex(subjects).fillna(np.nan)
        grand[label]    = per_subj[label].mean()

    # Grand mean bars: 
    grand_bar_width = 0.4
    grand_spacing   = 0.4
    grand_center    = 0.0

    grand_positions = np.array([
        grand_center + (i - (n_conds - 1) / 2) * grand_spacing
        for i in range(n_conds)
    ])

    # Subject bars
    subj_bar_width = 0.05
    subj_spacing   = 0.05
    subj_gap       = 0.1
    subj_start     = grand_positions[-1] + 0.6

    subj_centers = []
    for s in range(n_subj):
        center = subj_start + s * (n_conds * subj_spacing + 0.1)
        subj_centers.append(center)

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * n_subj + 6), 7))

    # Draw grand mean bars
    for i, (label, colour) in enumerate(zip(labels, colours)):
        pos = grand_positions[i]
        ax.bar(pos, grand[label], width=grand_bar_width,
               color=colour, alpha=0.85, edgecolor="white", label=label, zorder=2)
        ax.text(pos, grand[label] + 0.01, f"{grand[label]:.2f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.text(grand_center, -0.08, "All participants",
            ha="center", va="top", fontsize=12, fontweight="bold",
            transform=ax.get_xaxis_transform())

    # Draw per-subject bars
    for s, (subject, center) in enumerate(zip(subjects, subj_centers)):
        for i, (label, colour) in enumerate(zip(labels, colours)):
            pos = center + (i - (n_conds - 1) / 2) * subj_spacing
            val = per_subj[label][subject]
            if not np.isnan(val):
                ax.bar(pos, val, width=subj_bar_width,
                       color=colour, alpha=0.6, edgecolor="white", zorder=2)

        ax.text(center, -0.08, subject,
                ha="center", va="top", fontsize=7, rotation=30,
                transform=ax.get_xaxis_transform())

    # T-tests on grand mean bars
    pairs_list = [(0, 1), (1, 2), (0, 2)]
    y_base     = max(grand.values()) + 0.07
    step       = 0.07

    for k, (ci, cj) in enumerate(pairs_list):
        la, lb = labels[ci], labels[cj]
        vals_a = per_subj[la].dropna()
        vals_b = per_subj[lb].dropna()
        shared = vals_a.index.intersection(vals_b.index)
        if len(shared) < 2:
            continue
        t, p = stats.ttest_rel(vals_a[shared], vals_b[shared])
        if p >= 0.05:
            continue

        stars = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
        y  = y_base + k * step
        x1 = grand_positions[ci]
        x2 = grand_positions[cj]

        ax.plot([x1, x1, x2, x2], [y - 0.01, y, y, y - 0.01],
                color="black", linewidth=1.2)
        ax.text((x1 + x2) / 2, y + 0.005, stars,
                ha="center", va="bottom", fontsize=12)

    ax.set_xlim(grand_positions[0] - grand_bar_width,
                subj_centers[-1] + n_conds * subj_spacing)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Mean exact accuracy")
    ax.set_title("ASR accuracy comparison: no enhancement vs enhanced vs enhanced + prompt", fontsize = 24)
    ax.set_xticks([])
    ax.legend(loc="upper right", bbox_to_anchor=(0.92, 0.98))
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    os.makedirs(dir_info["figures"], exist_ok=True)
    plt.savefig(os.path.join(dir_info["figures"], "wer_comparison.png"), dpi=150, bbox_inches="tight")
    print("Saved: wer_comparison.png")
    plt.show()

def print_wer_ttest_table(data_annotated_clean, ann_info, dir_info, asr_configs, method="string"):
    from scipy import stats
    import itertools

    all_pairs = {}
    for cfg in asr_configs:
        asr_df = pd.read_csv(
            os.path.join(cfg["dir"], cfg["file"]), sep=",", encoding="latin-1"
        )
        asr_info_local = {
            "filename": "filename",
            "word":     "word_chosen",
            "time":     "start_time",
            "prob":     "prob_chosen",
        }
        _, asr_clean = clean_dataframes(data_annotated, asr_df, ann_info, asr_info_local)
        pairs = compute_pairs(data_annotated_clean, asr_clean, ann_info, asr_info_local, method=method)
        all_pairs[cfg["label"]] = pairs

    labels = [cfg["label"] for cfg in asr_configs]
    subjects = sorted(set().union(*[set(df["subject_id"]) for df in all_pairs.values()]))

    per_subj = {}
    grand    = {}
    for label in labels:
        df = all_pairs[label]
        df_valid = df[df["exact"].notna() & df["annotated"].ne("⟨INAUDIBLE⟩")].copy()
        df_valid["exact"] = df_valid["exact"].astype(float)
        per_subj[label] = df_valid.groupby("subject_id")["exact"].mean().reindex(subjects).fillna(np.nan)
        grand[label]    = per_subj[label].mean()

    records = []
    for la, lb in itertools.combinations(labels, 2):
        vals_a = per_subj[la].dropna()
        vals_b = per_subj[lb].dropna()
        shared = vals_a.index.intersection(vals_b.index)

        mean_a = grand[la]
        mean_b = grand[lb]
        sd_a   = per_subj[la].std()
        sd_b   = per_subj[lb].std()
        diff   = mean_a - mean_b

        if len(shared) < 2:
            records.append({
                "Condition A":  la,
                "Condition B":  lb,
                "Mean A (SD)":  f"{mean_a:.3f} ({sd_a:.3f})",
                "Mean B (SD)":  f"{mean_b:.3f} ({sd_b:.3f})",
                "Mean diff":    round(diff, 4),
                "t":            None,
                "df":           None,
                "p":            None,
                "Significant":  "—",
                "n (shared)":   len(shared),
            })
            continue

        t, p = stats.ttest_rel(vals_a[shared], vals_b[shared])
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

        records.append({
            "Condition A":  la,
            "Condition B":  lb,
            "Mean A (SD)":  f"{mean_a:.3f} ({sd_a:.3f})",
            "Mean B (SD)":  f"{mean_b:.3f} ({sd_b:.3f})",
            "Mean diff":    round(diff, 4),
            "t":            round(t, 4),
            "df":           len(shared) - 1,
            "p":            round(p, 4),
            "Significant":  stars,
            "n (shared)":   len(shared),
        })

    results_df = pd.DataFrame(records)

    print("\n── WER comparison t-test results ──────────────────────────────────────")
    print(results_df.to_string(index=False))
    print()

    out_path = os.path.join(dir_info["figures"], "wer_ttest_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return results_df

# Set direectories etc.

ANN_INFO = {
    "file":     "annotation(Merged).csv",
    "dir":      "../data_phonetic",
    "filename": "Filename",
    "word":     "Estimated Lexical Match",
    "time":     "Utterance start time",
    "id":       "ID",
}

ASR_INFO = {
    "file":     "ASR_transcription_alldata.csv",
    "dir":      "../output",
    "filename": "filename",
    "word":     "word_chosen",
    "time":     "start_time",
    "prob":     "prob_chosen",
}

DIR_INFO = {
    "figures": "../figures",
}

ASR_CONFIGS = [
    {"label": "No enhancement",       "dir": "../output", "file": "ASR_transcription_alldata.csv"},
    {"label": "Enhanced",             "dir": "../output", "file": "ASR_transcription_alldata_enhanced.csv"},
    {"label": "Enhanced + prompt",    "dir": "../output", "file": "ASR_transcription_alldata_enhanced_wprompt.csv"},
]


os.makedirs(DIR_INFO["figures"], exist_ok=True)

data_annotated, data_asr             = load_data(ANN_INFO, ASR_INFO)
data_annotated_clean, data_asr_clean = clean_dataframes(data_annotated, data_asr, ANN_INFO, ASR_INFO)

# Compare all three alignment methods on all the data to shose what we like 
#all_dfs = plot_alignment_comparison(data_annotated_clean, data_asr_clean, ANN_INFO, ASR_INFO, DIR_INFO)

# chose alignment comparison: "string", "greedy", or "hungarian"
ALIGNMENT_METHOD = "string"
#pairs_df = all_dfs[ALIGNMENT_METHOD]

#plot_timing(data_annotated_clean, data_asr_clean, ANN_INFO, ASR_INFO, DIR_INFO)
#plot_accuracy(pairs_df, DIR_INFO)
#plot_prob_correlations(pairs_df, DIR_INFO)
#plot_exact_boxplot(pairs_df, DIR_INFO)
#plot_confusion_matrices(pairs_df, DIR_INFO)
#plot_participant_accuracy(pairs_df, DIR_INFO)
plot_wer_comparison(data_annotated_clean, ANN_INFO, DIR_INFO, ASR_CONFIGS, method=ALIGNMENT_METHOD)
ttest_df = print_wer_ttest_table(data_annotated_clean, ANN_INFO, DIR_INFO, ASR_CONFIGS, method=ALIGNMENT_METHOD)
