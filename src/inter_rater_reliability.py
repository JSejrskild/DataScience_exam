# Here i test the inter-rater reliability for two participants annotated by Cboth of us

from Levenshtein import distance
import pandas as pd
import os
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay


def span_similarity(a, b):
    max_len = max(len(a), len(b))
    return 1 - distance(a, b) / max_len if max_len > 0 else 1.0

def load_data(files, filename1, filename2):
    annotation1 = pd.read_csv(
        os.path.join(files, filename1), sep=";", encoding="latin-1"
    )
    annotation2 = pd.read_csv(
        os.path.join(files, filename2), sep=";", encoding="latin-1"
    )
    return annotation1, annotation2

# Clean annotation files
def clean_dataframes(annotation1, annotation2):

    def clean_words(series):
        series = series.fillna("").astype(str).str.lower().str.strip()
        series = series.str.replace(r"[^a-z0-9æøå'\-]", "", regex=True)
        series = series.replace("", "⟨INAUDIBLE⟩")
        return series

    ann1 = annotation1.copy()
    for col in ann1.select_dtypes(include="object").columns:
        ann1[col] = ann1[col].fillna("").astype(str).str.lower().str.strip()

    ann1["Utterance start time"] = pd.to_numeric(ann1["Utterance start time"], errors="coerce")
    ann1["Estimated Lexical Match"] = clean_words(ann1["Estimated Lexical Match"])
    
    ann2 = annotation2.copy()
    for col in ann2.select_dtypes(include="object").columns:
        ann2[col] = ann2[col].fillna("").astype(str).str.lower().str.strip()

    ann2["Utterance start time"] = pd.to_numeric(ann2["Utterance start time"], errors="coerce")
    ann2["Estimated Lexical Match"] = clean_words(ann2["Estimated Lexical Match"])
    ann2["Filename"] = ann2["Filename"].str.removesuffix(".wav")
    
    return ann1, ann2


# Alignment because we may hear different numbers of words - A lot .. 
def align_annotators(annotation1_clean, annotation2_clean):
    all_aligned = []

    for fname, grp1 in annotation1_clean.groupby("Filename"):
        grp2 = annotation2_clean[annotation2_clean["Filename"] == fname]

        if grp2.empty:
            print(f"Skipping {fname}: not found in annotation2")
            continue

        aligned = _align_annotators_single(
            grp1["Estimated Lexical Match"].tolist(),
            grp1["Utterance start time"].tolist(),
            grp2["Estimated Lexical Match"].tolist(),
            grp2["Utterance start time"].tolist(),
        )
        aligned["Filename"] = fname
        

        aligned["similarity"] = aligned.apply(
            lambda r: span_similarity(r["ann1_word"], r["ann2_word"])
            if r["ann1_word"] != "⟨INAUDIBLE⟩" and r["ann2_word"] != "⟨INAUDIBLE⟩"
            else np.nan,
            axis=1,)
        all_aligned.append(aligned)

    return pd.concat(all_aligned, ignore_index=True)

# It is a function that goes for each filename - so we dont run it but it is used in the alignment function above
def _align_annotators_single(ann1_words, ann1_times, ann2_words, ann2_times):
    ann1_to_ann2 = {}
    matcher = SequenceMatcher(None, ann1_words, ann2_words, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "replace"):
            for i, j in zip(range(i1, i2), range(j1, j2)):
                ann1_to_ann2[i] = j

    rows = []
    for i, (word1, time1) in enumerate(zip(ann1_words, ann1_times)):
        if i in ann1_to_ann2:
            j = ann1_to_ann2[i]
            word2 = ann2_words[j]
            time2 = ann2_times[j]
        else:
            word2 = "⟨INAUDIBLE⟩"
            time2 = None

        rows.append({
            "ann1_word": word1,
            "ann1_time": time1,
            "ann2_word": word2,
            "ann2_time": time2,
            "time_diff": abs(time1 - time2) if time2 is not None else None,
        })

    matched_ann2 = set(ann1_to_ann2.values())
    for j, (word2, time2) in enumerate(zip(ann2_words, ann2_times)):
        if j not in matched_ann2:
            rows.append({
                "ann1_word": "⟨INAUDIBLE⟩",
                "ann1_time": None,
                "ann2_word": word2,
                "ann2_time": time2,
                "time_diff": None,
            })

    return pd.DataFrame(rows)

# Plot inter-rater reliability
def plot_irr(aligned, output_dir=None):

    for fname, group in aligned.groupby("Filename"):

        display = group[["ann1_word", "ann2_word", "similarity"]].copy()
        display["similarity"] = display["similarity"].apply(
            lambda v: f"{v:.2f}" if pd.notna(v) else "—"
        )
        display.columns = ["Annotator 1", "Annotator 2", "Similarity"]
        display = display.reset_index(drop=True)

        sim_col = list(display.columns).index("Similarity")
        cell_colours = []

        for _, row in display.iterrows():
            colours = [(1.0, 1.0, 1.0)] * len(display.columns)

            if row["Annotator 1"] == "⟨INAUDIBLE⟩" and row["Annotator 2"] == "⟨INAUDIBLE⟩":
                pass
            elif row["Annotator 1"] == row["Annotator 2"]:
                colours[0] = (0.7, 1.0, 0.7)
                colours[1] = (0.7, 1.0, 0.7)
            elif row["Annotator 1"] == "⟨INAUDIBLE⟩" or row["Annotator 2"] == "⟨INAUDIBLE⟩":
                colours[0] = (1.0, 0.9, 0.6)
                colours[1] = (1.0, 0.9, 0.6)
            else:
                colours[0] = (1.0, 0.7, 0.7)
                colours[1] = (1.0, 0.7, 0.7)

            try:
                v = float(row["Similarity"])
                colours[sim_col] = (1.0, 1.0, 1.0) if np.isnan(v) else (0.5 + 0.5 * (1 - v), 0.5 + 0.5 * v, 0.5)
            except (ValueError, TypeError):
                colours[sim_col] = (1.0, 1.0, 1.0)

            cell_colours.append(colours)

        n_match = (group["ann1_word"] == group["ann2_word"]).sum()
        n_total = len(display)

        fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * (n_total + 2))))
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
        ax.set_title(f"Inter-rater agreement — {fname} ({n_match}/{n_total} exact matches)", pad=14)
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            safe = fname.replace("/", "_").replace("\\", "_")
            plt.savefig(os.path.join(output_dir, f"irr_{safe}.png"), dpi=150, bbox_inches="tight")
            print(f"Saved: irr_{safe}.png")

        plt.show()

def mean_similarity(aligned, output_dir=None):
    agreements = []
    for _, row in aligned.iterrows():
        w1, w2 = row["ann1_word"], row["ann2_word"]
        if w1 == "⟨INAUDIBLE⟩" and w2 == "⟨INAUDIBLE⟩":
            agreements.append(1.0)
        elif w1 == "⟨INAUDIBLE⟩" or w2 == "⟨INAUDIBLE⟩":
            agreements.append(0.0)
        else:
            sim = span_similarity(w1, w2)
            agreements.append(sim if pd.notna(sim) else 0.0)

    mean = np.mean(agreements)
    print(f"Mean similarity: {mean:.3f}  (n={len(agreements)})")

    plt.figure(figsize=(7, 4))
    plt.hist(agreements, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
    plt.axvline(mean, color="red", linestyle="--", label=f"Mean = {mean:.3f}")
    plt.xlabel("Similarity")
    plt.ylabel("Count")
    plt.title("Distribution of inter-rater similarity scores")
    plt.legend()
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "similarity_distribution.png"), dpi=150, bbox_inches="tight")
        print("Saved: similarity_distribution.png")

    plt.show()
    return {"mean_similarity": round(mean, 3), "n": len(agreements)}


def cohens_kappa(aligned, output_dir=None, threshold=0.85):
    labels1, labels2 = [], []
    groups = [] 

    for _, row in aligned.iterrows():
        w1, w2 = row["ann1_word"], row["ann2_word"]
        if w1 == "⟨INAUDIBLE⟩" and w2 == "⟨INAUDIBLE⟩":
            labels1.append("inaudible")
            labels2.append("inaudible")
            groups.append("inaudible")
        elif w1 == "⟨INAUDIBLE⟩":
            labels1.append("inaudible")
            labels2.append("word")
            groups.append("inaudible")
        elif w2 == "⟨INAUDIBLE⟩":
            labels1.append("word")
            labels2.append("inaudible")
            groups.append("inaudible")
        else:
            sim = span_similarity(w1, w2)
            agreed = sim >= threshold if pd.notna(sim) else False
            labels1.append("agree" if agreed else "disagree")
            labels2.append("agree" if agreed else "disagree")
            groups.append("words")

    kappa = cohen_kappa_score(labels1, labels2)
    print(f"Cohen's Kappa: {kappa:.3f}  (threshold={threshold}, n={len(labels1)})")

    # Splitting the data into words that both could hear or words that were not fully audioable 
    def get_group(g):
        return [(l1, l2) for l1, l2, grp in zip(labels1, labels2, groups) if grp == g]

    words_pairs    = get_group("words")       # agree / disagree
    inaud_pairs    = get_group("inaudible")   # inaudible / word

    # Have two matricis
    def make_cm(pairs, label_set):
        l1s, l2s = zip(*pairs) if pairs else ([], [])
        return confusion_matrix(list(l1s), list(l2s), labels=label_set)

    cm_words = make_cm(words_pairs, ["agree", "disagree"])
    cm_inaud = make_cm(inaud_pairs, ["inaudible", "word"])
    
    
    # Plot both matrixes 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, cm, label_set, title in [
        (axes[0], cm_words, ["Agree on word", "Disagree on word"],  "Fully audible words"),
        (axes[1], cm_inaud, ["Inaudible - no annotation", "Annotated a word"],  "Partially/fully inaudible words"),
    ]:
        disp = ConfusionMatrixDisplay(cm, display_labels=label_set)
        disp.plot(cmap="Blues", ax=ax, colorbar=False)

        disp.ax_.xaxis.tick_bottom()
        disp.ax_.xaxis.set_label_position("bottom")

        disp.ax_.set_xlabel("Annotator 2", labelpad=10)
        disp.ax_.set_ylabel("Annotator 1")
        disp.ax_.set_title(title, pad=14)

        plt.setp(disp.ax_.get_xticklabels(), rotation=30, ha="right", fontstyle="italic")
        plt.setp(disp.ax_.get_yticklabels(), fontstyle="italic")

    fig.suptitle(f"Inter-rater confusion matrices  (κ = {kappa:.3f}, threshold ≥ {threshold})", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "kappa_confusion.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.show()
    return {"kappa": round(kappa, 3), "threshold": threshold, "n": len(labels1)}


# Directories
files = "../inter_rater_reliability"
filename1 = "annotation(Inter-rater-C).csv"
filename2 = "annotation(Inter-rater-J).csv"

annotation1, annotation2 = load_data(files, filename1, filename2)
annotation1_clean, annotation2_clean = clean_dataframes(annotation1, annotation2)
aligned = align_annotators(annotation1_clean, annotation2_clean)
#plot_irr(aligned, output_dir="../inter_rater_reliability")

mean_similarity(aligned, output_dir="../inter_rater_reliability")
#cohens_kappa(aligned, output_dir="../inter_rater_reliability")

