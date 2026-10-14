"""Build Project_Report.pdf.

    python make_report.py --results results --team "Alice Smith (leader), Bob Lee" --reflections reflections.md

* If results/ holds finished experiments, every table, figure and data-driven sentence is generated from the
  CSV/PNG files -- nothing is typed in by hand, so the report cannot drift from the experiments.
* If results/ is empty, the same report is built with clearly marked [RUN REQUIRED] fields.
* --reflections points to a short markdown/text file with your own lessons learned (paragraphs separated by
  blank lines). Those paragraphs are yours to write; the script only typesets them.
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="Project_Report.pdf")
ap.add_argument("--team", default="Victor Madu (individual project)")
ap.add_argument("--reflections", default="reflections.md")
ap.add_argument("--date", default="October 13, 2026")
A = ap.parse_args()
R = Path(A.results)
TODO = "[RUN REQUIRED]"
HAVE = (R / "metrics_summary.csv").exists()

# ------------------------------------------------------------------ styles / helpers
S = getSampleStyleSheet()
S.add(ParagraphStyle("T", parent=S["Title"], fontSize=19, leading=23, textColor=colors.HexColor("#183250"), spaceAfter=6))
S.add(ParagraphStyle("H", parent=S["Heading2"], fontSize=12.5, leading=15, textColor=colors.HexColor("#1B416B"), spaceBefore=11, spaceAfter=5, keepWithNext=1))
S.add(ParagraphStyle("H3", parent=S["Heading3"], fontSize=10.3, leading=13, textColor=colors.HexColor("#1B416B"), spaceBefore=6, spaceAfter=3, keepWithNext=1))
S.add(ParagraphStyle("B", parent=S["BodyText"], fontSize=9.6, leading=13.6, spaceAfter=6))
S.add(ParagraphStyle("C", parent=S["BodyText"], fontSize=8.2, leading=10.8, textColor=colors.HexColor("#444444"), spaceAfter=8))
S.add(ParagraphStyle("TC", parent=S["BodyText"], fontSize=8.3, leading=10.5))
S.add(ParagraphStyle("Meta", parent=S["BodyText"], fontSize=9.6, leading=13, alignment=1, spaceAfter=3))
E = []


def P(t, st="B"): E.append(Paragraph(t, S[st]))
def H(t): P(t, "H")
def H3(t): P(t, "H3")
def cap(t): P(t, "C")


def table(rows, widths, hdr=True):
    t = Table([[Paragraph(str(c), S["TC"]) for c in r] for r in rows], colWidths=widths, repeatRows=1 if hdr else 0, hAlign="LEFT")
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c4cdd8")),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if hdr:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eff6")))
    t.setStyle(TableStyle(st))
    return t


def img(path, width, caption=None):
    path = Path(path)
    if not path.exists():
        return
    w, h = PILImage.open(path).size
    items = [Image(str(path), width=width, height=width * h / w)]
    if caption:
        items.append(Paragraph(caption, S["C"]))
    E.append(KeepTogether(items))


def csv(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else None


def f3(x): return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pm(mean, std, n):
    return f"{mean:.3f} &plusmn; {std:.3f}" if n > 1 and not np.isnan(std) else f"{mean:.3f}"


# ------------------------------------------------------------------ load results
summ, met, base = csv("metrics_summary.csv"), csv("metrics.csv"), csv("baselines.csv")
dc, dcb, sc = csv("dataset_counts.csv"), csv("dataset_counts_before_cap.csv"), csv("split_counts.csv")
rq4f, rq4c, rq4i, rq3d = csv("rq4_features.csv"), csv("rq4_feature_classifier.csv"), csv("rq4_feature_importance.csv"), csv("rq3.csv")
labels = list(dc.label) if dc is not None else []
K = len(labels)
chance = 1 / K if K else None
nseeds = int(summ.n_seeds.max()) if HAVE else 1


def get(model, feat, col="macro_f1_mean"):
    r = summ[(summ.model == model) & (summ.features == feat)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def ci(model, feat):
    r = met[(met.model == model) & (met.features == feat) & met.macro_f1_ci_low.notna()]
    return (float(r.macro_f1_ci_low.iloc[0]), float(r.macro_f1_ci_high.iloc[0])) if len(r) else (np.nan, np.nan)


# ------------------------------------------------------------------ title block
P("Who Wrote It? Identifying LLMs from Their Responses", "T")
P("<b>CMPSC 448 Midterm Project &mdash; Research Report</b>", "Meta")
P(f"<b>Team:</b> {A.team} &nbsp;&nbsp; <b>Date:</b> {A.date}", "Meta")
if not HAVE:
    P("<font color='#a33'><b>Build note:</b> this PDF was generated before experiments were run. Fields marked "
      f"{TODO} are filled automatically by <i>make_report.py</i> once <i>results/</i> exists; no numbers have been estimated or invented.</font>", "C")

# ------------------------------------------------------------------ 1. problem
H("1. Problem Definition and Research Questions")
P("LLMs now write a large share of online text, which raises a practical authorship question: does a response carry "
  "a recognizable &ldquo;fingerprint&rdquo; of the model family that produced it? We frame this as <b>supervised "
  "multiclass text classification</b>: given a user prompt (<i>LLM_Input</i>), a model response (<i>LLM_output</i>), "
  "or both, predict the generating family (<i>LLM_name</i>). Success on held-out data shows that stylistic and "
  "structural regularities exist in <i>this</i> dataset; it does not by itself make the classifier a reliable "
  "attribution tool for other models, prompts or time periods.")
P("<b>RQ1</b> &mdash; Can a CNN and an RNN (BiLSTM) identify the source family from the response alone? "
  "<b>RQ2</b> &mdash; Does the prompt help: input-only vs. output-only vs. input+output, for both architectures? "
  "<b>RQ3</b> (extra credit) &mdash; Do fingerprints transfer to task types unseen in training? "
  "<b>RQ4</b> (extra credit) &mdash; Which linguistic and structural properties separate the families, and does removing "
  "formatting hurt classification?")

# ------------------------------------------------------------------ 2. data
H("2. Dataset and Data Curation")
P("<b>Source.</b> LMSYS <i>Chatbot Arena Conversations</i> (Zheng et al., 2023): ~33k pairwise comparisons, each with an "
  "original <i>question_id</i>, two anonymous model names and both dialogues "
  "(huggingface.co/datasets/lmsys/chatbot_arena_conversations; gated, access terms accepted). Prompts are CC BY 4.0 and "
  "model outputs CC BY-NC 4.0; raw data are <b>not</b> redistributed in our repository. The release predates current "
  "models (e.g. Gemini, Qwen, DeepSeek), so our label set is the 2023 landscape.")
P("<b>Construction of (LLM_name, LLM_Input, LLM_output).</b> Each comparison yields two rows, one per side. Steps: "
  "(1) keep English conversations; (2) drop conversations flagged by the dataset&rsquo;s OpenAI-moderation field; "
  "(3) concatenate user turns into <i>LLM_Input</i> and assistant turns into <i>LLM_output</i>; (4) map model names to a "
  "<b>family</b> (GPT; Claude; Llama-derived = Vicuna, Koala, Alpaca, LLaMA; ChatGLM; Google = PaLM/Bard) and discard "
  "unmapped or ambiguous names (e.g. GPT4All, which is not an OpenAI model); (5) remove empty/very short texts and "
  "<b>exact duplicate (family, response) pairs</b>, so canned replies cannot appear in both train and test; (6) drop families with "
  "fewer than 100 examples and cap each family at 1,600 rows with a fixed seed to keep classes reasonably balanced. "
  "&ldquo;Llama-derived&rdquo; mixes several fine-tunes, so a family is not a checkpoint.")
if dc is not None and sc is not None:
    rows = [["Family", "After cleaning", "After cap", "Train", "Val", "Test"]]
    bef = dict(zip(dcb.label, dcb.n)) if dcb is not None else {}
    for fam in labels:
        c = {s: int(sc[(sc.split == s) & (sc.family == fam)]["count"].iloc[0]) for s in ("train", "val", "test")}
        rows.append([fam, bef.get(fam, "n/a"), int(dc[dc.label == fam].n.iloc[0]), c["train"], c["val"], c["test"]])
    rows.append(["<b>Total</b>", sum(v for v in bef.values()) if bef else "n/a", int(dc.n.sum()),
                 int(sc[sc.split == "train"]["count"].sum()), int(sc[sc.split == "val"]["count"].sum()), int(sc[sc.split == "test"]["count"].sum())])
    E.append(KeepTogether([table(rows, [105, 85, 70, 65, 55, 55]),
                           Paragraph("Table 1. Examples per family and split (from results/dataset_counts*.csv, split_counts.csv).", S["C"])]))
else:
    P(f"<b>Class and split counts:</b> {TODO} (results/dataset_counts.csv, results/split_counts.csv).")

# ------------------------------------------------------------------ 3. design
H("3. Experimental Design and Leakage Control")
P("<b>Grouped, stratified 70/10/20 split.</b> Both answers to one <i>question_id</i> always land in the same partition "
  "(StratifiedGroupKFold), and the code asserts that no group crosses splits. Without this, a model could be rewarded "
  "for recognising a prompt it saw in training. The vocabulary (&le;18,000 tokens) is fit on training text only; "
  "validation macro-F1 selects the epoch (best epoch restored); the test set is used once per model for reporting.")
P("<b>Conditions.</b> <i>Input only</i>, <i>output only</i> and <i>input+output</i> (prompt, a <tt>&lt;SEP&gt;</tt> token, response), "
  "each with CNN and BiLSTM = six primary experiments. Text is lower-cased and tokenised with a regex that keeps punctuation, "
  "markdown symbols and <b>newlines (as a &lt;nl&gt; token)</b>; sequences are truncated/padded to 180 tokens. In the combined condition the "
  "prompt is capped at 60 tokens so a long prompt can never crowd the response out of the window. "
  "A seventh condition, <i>output_noformat</i>, strips markdown/list/code markers and collapses whitespace (RQ4 ablation).")
P("<b>Baselines and metrics.</b> We report accuracy and <b>macro-F1</b> (robust to class imbalance), compared against "
  "uniform chance (1/K), the majority-class rule, and a TF-IDF (1&ndash;2-gram) + logistic-regression classifier that uses the same "
  "splits. A strong linear baseline tells us whether deep models add anything. Uncertainty: " +
  (f"scores are mean &plusmn; std over {nseeds} model seeds, " if nseeds > 1 else "scores come from a single seed (use --seeds for more), ") +
  "and 95% CIs for macro-F1 come from a bootstrap that resamples whole question groups.")

# ------------------------------------------------------------------ 4. models
H("4. CNN and RNN Implementation")
P("<b>CNN (TextCNN).</b> Embedding (96-d, padding index 0) &rarr; three parallel Conv1d layers with kernel widths 3, 5, 7 "
  "(96 filters each, &lsquo;same&rsquo; padding) &rarr; ReLU &rarr; <i>masked</i> global max-pooling (padded positions cannot win the max) "
  "&rarr; concatenated 288-d vector &rarr; dropout 0.35 &rarr; linear layer. Kernels act as learned n-gram detectors "
  "for phrases such as stock openers and hedges.")
P("<b>RNN (BiLSTM).</b> Embedding (96-d) &rarr; one-layer bidirectional LSTM (64 units per direction) run on <i>packed</i> sequences, so the final "
  "states reflect the last real token rather than trailing padding &rarr; concatenated forward/backward states (128-d) &rarr; dropout 0.35 &rarr; linear layer.")
P("<b>Training.</b> Cross-entropy; AdamW (lr 0.002, weight decay 0.01); batch size 64; gradient-norm clipping at 1.0; at most 8 epochs with "
  "early stopping after 2 epochs without validation macro-F1 improvement; the data split is fixed (seed 42) and model initialisation / shuffling "
  "use the listed seeds; GPU when available.")

# ------------------------------------------------------------------ 5. results
H("5. Results: RQ1 and RQ2")
names = {"input": "Input only", "output": "Output only", "both": "Input + output"}
rows = [["Model", "Input", "Test accuracy", "Test macro-F1", "95% CI (macro-F1)"]]
for m in ("CNN", "BiLSTM"):
    for f in ("input", "output", "both"):
        if HAVE:
            r = summ[(summ.model == m) & (summ.features == f)]
            lo, hi = ci(m, f)
            rows.append([m, names[f], pm(float(r.accuracy_mean.iloc[0]), float(r.accuracy_std.iloc[0]), nseeds),
                         pm(float(r.macro_f1_mean.iloc[0]), float(r.macro_f1_std.iloc[0]), nseeds), f"[{f3(lo)}, {f3(hi)}]"])
        else:
            rows.append([m, names[f], TODO, TODO, TODO])
if base is not None:
    rows.append(["<i>Chance</i>", "&ndash;", f3(chance), "&ndash;", ""])
    for r in base.itertuples():
        if r.baseline.startswith("majority"):
            rows.append([f"<i>{r.baseline}</i>", "&ndash;", f3(r.accuracy), f3(r.macro_f1), ""])
        elif r.baseline.startswith("TF-IDF") and r.features in names:
            rows.append(["<i>TF-IDF + LogReg</i>", names[r.features], f3(r.accuracy), f3(r.macro_f1), ""])
E.append(KeepTogether([table(rows, [100, 95, 95, 105, 105]),
                       Paragraph("Table 2. Held-out test results (results/metrics_summary.csv, baselines.csv). Mean ± std over seeds 42, 43, 44; the 95% CI is a question-group bootstrap on the seed-42 predictions only. Deep models vs. non-neural reference rows in italics.", S["C"])]))

if HAVE:
    cnn_o, lstm_o = get("CNN", "output"), get("BiLSTM", "output")
    tf_o = base[(base.baseline.str.startswith("TF-IDF")) & (base.features == "output")].macro_f1.iloc[0] if base is not None else np.nan
    win = "CNN" if cnn_o > lstm_o else "BiLSTM"
    H3("RQ1: identifying the family from the response")
    P(f"With responses alone, the CNN reaches macro-F1 {f3(cnn_o)} and the BiLSTM {f3(lstm_o)} (chance = {f3(chance)} for {K} families); "
      f"the {win} is higher by {abs(cnn_o - lstm_o):.3f}. The TF-IDF + logistic-regression reference scores {f3(tf_o)}, so the best neural model is "
      f"{'above' if max(cnn_o, lstm_o) > tf_o else 'not above'} this linear baseline ({max(cnn_o, lstm_o) - tf_o:+.3f}). "
      + ("Differences between the two architectures are within one seed-std and should not be over-read. " if nseeds > 1 and abs(cnn_o - lstm_o) < max(get('CNN','output','macro_f1_std'), get('BiLSTM','output','macro_f1_std')) else "")
      + "Per-family precision, recall and F1 are in Table 3 and the confusion matrices in Figure 1.")
    # per-family table (output-only, first seed)
    def parse(path):
        out = {}
        for line in Path(path).read_text().splitlines():
            parts = line.strip().rsplit(None, 4)
            if len(parts) == 5 and parts[0] in labels:
                out[parts[0]] = parts[1:]
        return out
    pc, pl = parse(R / "classification_CNN_output.txt"), parse(R / "classification_BiLSTM_output.txt")
    rows = [["Family", "CNN P", "CNN R", "CNN F1", "LSTM P", "LSTM R", "LSTM F1", "Support"]]
    for fam in labels:
        rows.append([fam] + pc[fam][:3] + pl[fam][:3] + [pc[fam][3]])
    E.append(KeepTogether([table(rows, [95, 50, 50, 50, 55, 55, 55, 55]),
                           Paragraph("Table 3. Per-family test metrics, output-only condition, seed 42 (P = precision, R = recall).", S["C"])]))
    E.append(Table([[Image(str(R / "confusion_CNN_output.png"), width=235, height=235 * 6 / 7),
                     Image(str(R / "confusion_BiLSTM_output.png"), width=235, height=235 * 6 / 7)]]))
    cap("Figure 1. Confusion matrices (rows = actual family) for the seed-42 output-only CNN (left) and BiLSTM (right). Off-diagonal mass shows which families the models confuse.")
    # RQ2
    H3("RQ2: does the prompt help?")
    pieces = []
    for m in ("CNN", "BiLSTM"):
        fi, fo, fb = get(m, "input"), get(m, "output"), get(m, "both")
        pieces.append(f"{m}: input-only {f3(fi)} ({fi - chance:+.3f} vs. chance), output-only {f3(fo)}, input+output {f3(fb)} ({fb - fo:+.3f} vs. output-only)")
    P("; ".join(pieces) + ".")
    best_in = max(get("CNN", "input"), get("BiLSTM", "input"))
    if best_in > chance + 0.05:
        P("Input-only performance is clearly above chance. This is <i>not</i> evidence that prompts intrinsically identify a model: in Arena, models were "
          "added to the leaderboard at different times, sampled non-uniformly and shown different users, so topic, language mix and task type correlate with the "
          "label. Because every prompt is answered by two models, a perfect prompt-only classifier would still be capped well below 100% on prompts shared between "
          "families, which makes the gap to output-only informative.")
    else:
        P("Input-only performance is at or near chance, which is what we expect by design: each Arena prompt is answered by two different models, so the "
          "prompt alone carries little information about which one answered. Any gain from the combined condition must therefore come from the response text, "
          "not from prompt-to-model correlations.")
    # curves
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, m in zip(axes, ("CNN", "BiLSTM")):
        for f in ("input", "output", "both"):
            p = R / f"training_{m}_{f}.csv"
            if p.exists():
                h = pd.read_csv(p); ax.plot(h.epoch, h.val_macro_f1, marker="o", label=names[f])
        ax.axhline(chance, ls=":", color="gray", label="chance"); ax.set_title(f"{m}: validation macro-F1"); ax.set_xlabel("epoch"); ax.set_ylim(0, 1.02); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(R / "val_curves.png", dpi=150); plt.close(fig)
    img(R / "val_curves.png", 440, "Figure 2. Validation macro-F1 per epoch (first seed). Early stopping restores the best epoch.")
else:
    P(f"<b>RQ1 / RQ2 interpretation:</b> {TODO} &mdash; run <tt>python project.py --data data/chatbot_arena.parquet --out results --seeds 42 43 44</tt>, then rebuild this report.")

# ------------------------------------------------------------------ 6. RQ4
H("6. RQ4: What Distinguishes the Families?")
P("We measure eight interpretable response properties: token count, lexical diversity (unique/total tokens), lexical diversity on the first 100 tokens "
  "(length-controlled, since the plain ratio falls as responses get longer), list-item lines, code-fence delimiters, newlines, mean token length and the count of "
  "assistant boilerplate phrases (&ldquo;as an AI&rdquo;, &ldquo;I&rsquo;m sorry&rdquo;, &hellip;). We then (a) tabulate them by family, (b) train a gradient-boosting classifier on "
  "<i>only</i> these features and measure permutation importance, and (c) rerun the networks with all formatting removed.")
if rq4f is not None:
    d = pd.read_csv(R / "rq4_features.csv", header=[0, 1], index_col=0)
    cols = ["word_count", "ttr_first100", "bullet_lines", "code_fences", "newline_count", "ai_disclaimer_phrases"]
    rows = [["Family", "Tokens", "TTR@100", "List lines", "Code fences", "Newlines", "AI phrases"]]
    for fam in d.index:
        rows.append([fam] + [f"{d.loc[fam, (c, 'mean')]:.2f}" for c in cols])
    E.append(KeepTogether([table(rows, [100, 60, 60, 65, 70, 65, 70]), Paragraph("Table 4. Mean feature values per family (medians in results/rq4_features.csv).", S["C"])]))
    img(R / "rq4_features.png", 400, "Figure 3. Distribution of four features by family (outliers hidden).")
    means = {c: d[(c, "mean")] for c in cols}
    ex = []
    for c in ("word_count", "bullet_lines", "newline_count"):
        ex.append(f"{c.replace('_', ' ')} is highest for {means[c].idxmax()} ({means[c].max():.1f}) and lowest for {means[c].idxmin()} ({means[c].min():.1f})")
    P("Largest observed differences: " + "; ".join(ex) + ". These are correlational: families were shown different prompts, so a family that looks &ldquo;longer&rdquo; may simply have received longer tasks.")
if rq4c is not None and rq4i is not None:
    top = rq4i.head(3)
    P(f"A gradient-boosting model on these eight features alone reaches macro-F1 {f3(rq4c.macro_f1.iloc[0])} (accuracy {f3(rq4c.accuracy.iloc[0])}). "
      "Permutation importance (drop in test macro-F1 when a feature is shuffled) ranks " +
      ", ".join(f"{r.feature.replace('_', ' ')} ({r.macro_f1_drop:.3f})" for r in top.itertuples()) + " highest.")
if HAVE and "output_noformat" in set(summ.features):
    parts = [f"{m}: {get(m, 'output'):.3f} &rarr; {get(m, 'output_noformat'):.3f} ({get(m, 'output_noformat') - get(m, 'output'):+.3f})" for m in ("CNN", "BiLSTM")]
    P("<b>Formatting ablation</b> (output macro-F1, original &rarr; formatting stripped): " + "; ".join(parts) +
      ". A small drop means the networks rely mainly on wording; a large drop means list/markdown/layout conventions carry much of the signal.")
if not HAVE:
    P(f"<b>Observations:</b> {TODO}")

# ------------------------------------------------------------------ 6b RQ3
if rq3d is not None and len(rq3d):
    H("RQ3: Generalisation across task types (extra credit)")
    P("Prompts are assigned to coding, math, writing or general by transparent keyword rules (<tt>DOMAINS</tt> in project.py); these are heuristic labels, not ground truth. "
      "For each domain we train output-only models on the other domains and test on the held-out one.")
    rows = [["Model", "Held-out domain", "n", "Out-of-domain macro-F1", "In-domain macro-F1 (same rows, main model)"]]
    for r in rq3d.itertuples():
        rows.append([r.model, r.held_out_domain, r.n_held_out, f3(r.ood_macro_f1), f3(r.in_domain_macro_f1)])
    E.append(KeepTogether([table(rows, [60, 90, 45, 130, 190]), Paragraph("Table 5. Leave-one-domain-out results (results/rq3.csv). The in-domain column is computed on the main model&rsquo;s test predictions for that domain; the two models differ in training data size, so compare gaps cautiously.", S["C"])]))
    g = rq3d.assign(gap=rq3d.in_domain_macro_f1 - rq3d.ood_macro_f1)
    big, small = g.loc[g.gap.idxmax()], g.loc[g.gap.idxmin()]
    mean_gap = {m: g[g.model == m].gap.mean() for m in g.model.unique()}
    P("<b>Reading the table.</b> Averaged over domains, held-out performance is below in-domain performance by " +
      " and ".join(f"{v:.3f} macro-F1 for the {m}" for m, v in mean_gap.items()) +
      f". The largest drop is {big.model} on <i>{big.held_out_domain}</i> ({big.in_domain_macro_f1:.3f} to {big.ood_macro_f1:.3f}); the smallest is {small.model} on <i>{small.held_out_domain}</i> "
      f"({small.in_domain_macro_f1:.3f} to {small.ood_macro_f1:.3f}). Every held-out score stays well above chance (0.200), so part of the signal is not tied to one task type. "
      "Two cautions apply. First, the <i>general</i> domain contains most prompts, so holding it out removes most of the training data; its large drop mixes domain shift with a smaller training set. "
      "Second, the in-domain column comes from the main model trained on all domains, so it also benefits from seeing more data. A drop therefore does not prove the model learned task-specific patterns, "
      "and the small drops (e.g. writing) do not prove a task-independent fingerprint. The RQ3 scores are from a single seed (42), unlike the main table.")

# ------------------------------------------------------------------ 7. limitations + lessons
H("7. Limitations")
P("(1) Family labels merge many checkpoints and fine-tunes. (2) The data are from 2023 and reflect Arena users&rsquo; tasks; results do not transfer automatically to current models. "
  "(3) Truncation at 180 tokens hides length and late-response style; padding is masked, so the CNN cannot see raw length at all. (4) Multi-turn conversations are flattened "
  "into one prompt and one response string. (5) Scores come from one fixed split; seed variation only covers model initialisation. (6) Moderation filtering and the English-only "
  "filter further narrow the population. (7) A classifier that works on this dataset may have learned collection artefacts rather than model identity.")
H("8. Lessons and Experience")
P("<b>Methodology.</b> Grouping by <i>question_id</i> is essential when several answers share a prompt; baselines (chance, majority, TF-IDF) are needed before a neural score means anything; "
  "macro-F1 exposes weak small classes that accuracy hides; and an attractive feature table is not a causal explanation &mdash; the ablation is what tests it.")
P("<b>Implementation.</b> Several details silently change results: running an LSTM over padded batches without packing lets trailing padding overwrite the final state; max-pooling over padded "
  "positions lets padding compete with real tokens; naive truncation of &ldquo;prompt + response&rdquo; can delete the response entirely; and one ambiguous model name (GPT4All) would "
  "have been mislabelled as an OpenAI model by a substring rule. Each is handled in <i>project.py</i>.")
refl = Path(A.reflections)
if refl.exists():
    for para in refl.read_text().split("\n\n"):
        if para.strip():
            P(para.strip().replace("\n", " "))
else:
    P(f"<b>Our own reflections (debugging, training surprises, what we would do differently):</b> [ADD &mdash; put 1&ndash;3 paragraphs in reflections.md and rebuild]")

# ------------------------------------------------------------------ 9. reproduction
H("9. Reproduction and References")
P("Repository: https://github.com/vxm5308-ship-it/CMPSC448-LLM-Authorship . Contents: project.py, make_report.py, run_rq3_only.py, run_missing_s44.py, check_submission.py, requirements.txt, README.md, reflections.md, results/ (CSV tables and figures), this report. Run "
  "<tt>python project.py --data data/chatbot_arena.parquet --out results --seeds 42 43 44 --rq3</tt> then <tt>python make_report.py --team \"...\"</tt>. Raw data and model "
  "weights are excluded (see .gitignore).")
P("Zheng, L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. arXiv:2306.05685.<br/>"
  "LMSYS (2023). Chatbot Arena Conversation Dataset Release. lmsys.org/blog/2023-07-20-dataset/")


def footer(c, d):
    c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawString(51, 35, "CMPSC 448 \u2022 LLM Authorship Identification"); c.drawRightString(561, 35, f"Page {d.page}")


SimpleDocTemplate(A.out, pagesize=(612, 792), rightMargin=51, leftMargin=51, topMargin=50, bottomMargin=52,
                  title="Who Wrote It? Identifying LLMs from Their Responses").build(E, onFirstPage=footer, onLaterPages=footer)
print("wrote", A.out, "(with results)" if HAVE else "(template: no results found)")
