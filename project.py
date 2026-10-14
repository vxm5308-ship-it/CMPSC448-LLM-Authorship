"""CMPSC 448 midterm: LLM authorship identification (CNN vs BiLSTM).

Usage
-----
    python project.py --data data/chatbot_arena.parquet --out results
    python project.py --data data/my_data.csv --out results      # columns: LLM_name,LLM_Input,LLM_output
    python project.py --data data/chatbot_arena.parquet --seeds 42 43 44 --rq3

What it produces in --out (everything the report needs; nothing is simulated):
    metrics.csv / metrics_summary.csv   per-seed and mean+-std test scores for each model x condition
    baselines.csv                       chance, majority-class, TF-IDF+LogReg per condition
    classification_*.txt, confusion_*.png, training_*.csv, predictions.csv
    rq4_*                               feature table, plots, feature-only classifier, permutation importance,
                                        and the "formatting removed" ablation (condition `output_noformat`)
    rq3.csv                             (optional, --rq3) leave-one-domain-out generalisation
    dataset_counts*.csv, split_counts.csv, run_config.json
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42  # fixes the data split; model seeds are set with --seeds


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- text processing
# Newlines are kept as an explicit <nl> token and "<SEP>" is a single token, so the networks can
# see paragraph/list structure (a known LLM stylistic signal) and the prompt/response boundary.
TOKEN_RE = re.compile(r"<sep>|\n|\b\w+\b|[^\w\s]", re.UNICODE)


def tokenize(text):
    return ["<nl>" if t == "\n" else t for t in TOKEN_RE.findall(str(text).lower())]


def strip_format(text):
    """Ablation: remove markdown/list/code formatting and collapse all whitespace (incl. newlines)."""
    t = re.sub(r"```[^\n]*\n?", " ", str(text))
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)
    t = re.sub(r"(?m)^\s*([-*\u2022]|\d+[.)])\s+", "", t)
    t = re.sub(r"\*+|`+", "", t)
    return re.sub(r"\s+", " ", t).strip()


def family(name):
    """Map an Arena model name to a broad family. Unknown/ambiguous names return None (excluded)."""
    s = str(name).lower()
    if "gpt4all" in s:  # a LLaMA/GPT-J fine-tune, not an OpenAI model: exclude rather than mislabel
        return None
    if "gpt" in s:
        return "GPT"
    if "claude" in s:
        return "Claude"
    if any(v in s for v in ("vicuna", "llama", "koala", "alpaca", "guanaco")):
        return "Llama-derived"
    if "chatglm" in s:
        return "ChatGLM"
    if any(v in s for v in ("palm", "bard", "gemini")):
        return "Google"
    if "qwen" in s:
        return "Qwen"
    if "deepseek" in s:
        return "DeepSeek"
    return None


# --------------------------------------------------------------------------- data loading
def extract_turns(conv):
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except (ValueError, TypeError):
            return []
    if not isinstance(conv, (list, tuple, np.ndarray)):
        return []
    return [x for x in conv if isinstance(x, dict)]


def is_flagged(mod):
    """OpenAI-moderation field can be a dict, a list/array of dicts, or a JSON string."""
    if isinstance(mod, str):
        try:
            mod = json.loads(mod)
        except ValueError:
            return False
    if isinstance(mod, dict):
        return bool(mod.get("flagged", False))
    if isinstance(mod, (list, tuple, np.ndarray)):
        return any(is_flagged(m) for m in mod)
    return False


def read_data(path, max_per_family=1600, min_per_family=100):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Download an authorized dataset or supply a CSV with LLM_name, LLM_Input, LLM_output. See README.md.")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        raw = pd.read_parquet(path)
    elif suffix == ".jsonl":
        raw = pd.read_json(path, lines=True)
    else:
        raise ValueError("Data must be .csv, .parquet, or .jsonl")

    if {"LLM_name", "LLM_Input", "LLM_output"}.issubset(raw.columns):
        df = raw.rename(columns={"LLM_name": "label", "LLM_Input": "prompt", "LLM_output": "response"}).copy()
        df["label"] = df["label"].map(lambda x: family(x) or str(x))
        if "question_id" in df.columns:
            df["group"] = df["question_id"].astype(str)
        else:  # no id: identical prompts must still stay in one split
            df["group"] = df["prompt"].astype(str).map(lambda s: hashlib.md5(s.encode()).hexdigest())
    elif {"model_a", "model_b", "conversation_a", "conversation_b", "question_id"}.issubset(raw.columns):
        rows = []
        for row in raw.itertuples(index=False):
            if hasattr(row, "language") and str(row.language).lower() not in ("en", "english"):
                continue
            if hasattr(row, "openai_moderation") and is_flagged(row.openai_moderation):
                continue
            for side in ("a", "b"):
                lab = family(getattr(row, "model_" + side))
                if lab is None:
                    continue
                turns = extract_turns(getattr(row, "conversation_" + side))
                prompt = "\n".join(str(t.get("content", "")) for t in turns if t.get("role") == "user")
                response = "\n".join(str(t.get("content", "")) for t in turns if t.get("role") == "assistant")
                rows.append({"label": lab, "prompt": prompt, "response": response, "group": str(row.question_id)})
        df = pd.DataFrame(rows)
    else:
        raise ValueError("Expected columns LLM_name,LLM_Input,LLM_output or the original Chatbot Arena schema")

    df = df.dropna(subset=["prompt", "response", "label"]).copy()
    df["prompt"] = df.prompt.astype(str)
    df["response"] = df.response.astype(str)
    df = df[(df.prompt.str.len() >= 8) & (df.response.str.len() >= 20)]
    # Identical (label, response) pairs, e.g. canned refusals, would otherwise appear in train AND test.
    df = df.drop_duplicates(subset=["label", "response"])
    raw_counts = df.label.value_counts()
    keep = raw_counts[raw_counts >= min_per_family].index
    df = df[df.label.isin(keep)].copy()
    if df.label.nunique() < 3:
        raise ValueError(f"Need >=3 families with >= {min_per_family} examples. Counts: {raw_counts.to_dict()}")
    df = pd.concat(
        [g.sample(min(len(g), max_per_family), random_state=SEED) for _, g in df.groupby("label")],
        ignore_index=True,
    )
    return df, raw_counts


def group_split(frame, test_frac, seed=SEED):
    """Stratified, group-aware split: no group (original question) is shared between the two parts."""
    sgkf = StratifiedGroupKFold(n_splits=round(1 / test_frac), shuffle=True, random_state=seed)
    a, b = next(sgkf.split(frame, frame.label, groups=frame.group))
    return frame.iloc[a].reset_index(drop=True), frame.iloc[b].reset_index(drop=True)


def make_splits(df):
    trainval, test = group_split(df, 0.2)
    train, val = group_split(trainval, 0.125)  # 0.8 * 0.125 = 10% of all data
    labels = sorted(df.label.unique())
    for name, frame in (("train", train), ("val", val), ("test", test)):
        missing = set(labels) - set(frame.label)
        if missing:
            raise ValueError(f"{name} split lacks {missing}; use more data")
    assert not (set(train.group) & set(val.group) | set(train.group) & set(test.group) | set(val.group) & set(test.group)), \
        "group leakage between splits"
    return train, val, test, labels


# --------------------------------------------------------------------------- tensors / models
def build_tokens(frame, mode, seq_len, ablate=False):
    """Token lists truncated to seq_len. In 'both' mode the prompt is capped at seq_len//3 tokens so a long
    prompt can never push the response out of the window."""
    prep = strip_format if ablate else (lambda s: s)
    if mode == "input":
        return [tokenize(prep(p))[:seq_len] for p in frame.prompt]
    if mode == "output":
        return [tokenize(prep(r))[:seq_len] for r in frame.response]
    p_len = seq_len // 3
    return [(tokenize(prep(p))[:p_len] + ["<sep>"] + tokenize(prep(r)))[:seq_len]
            for p, r in zip(frame.prompt, frame.response)]


def vocab_fit(token_lists, max_vocab=18000):
    count = Counter(t for toks in token_lists for t in toks)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in count.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


class TextSet(Dataset):
    def __init__(self, token_lists, labels, vocab, seq_len):
        ids = []
        for toks in token_lists:
            arr = [vocab.get(w, 1) for w in toks[:seq_len]]
            ids.append(arr + [0] * (seq_len - len(arr)))
        self.x = torch.tensor(ids, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class TextCNN(nn.Module):
    """Embedding -> parallel Conv1d (k=3,5,7) -> ReLU -> masked global max-pool -> dropout -> linear."""

    def __init__(self, vocab_size, n_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 96, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(96, 96, k, padding=k // 2) for k in (3, 5, 7)])
        self.drop = nn.Dropout(0.35)
        self.fc = nn.Linear(96 * 3, n_labels)

    def forward(self, x):
        mask = (x != 0).unsqueeze(1)  # padded positions must not contribute to the max
        e = self.emb(x).transpose(1, 2)
        feats = [torch.relu(conv(e)).masked_fill(~mask, 0.0).amax(dim=2) for conv in self.convs]
        return self.fc(self.drop(torch.cat(feats, dim=1)))


class TextLSTM(nn.Module):
    """Embedding -> packed BiLSTM (64/direction) -> concat final fwd/bwd states -> dropout -> linear."""

    def __init__(self, vocab_size, n_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 96, padding_idx=0)
        self.rnn = nn.LSTM(96, 64, num_layers=1, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(0.35)
        self.fc = nn.Linear(128, n_labels)

    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1).cpu()  # packing => final state ignores trailing padding
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(packed)
        return self.fc(self.drop(torch.cat((h[-2], h[-1]), dim=1)))


def evaluate(model, loader, device):
    model.eval()
    y, p = [], []
    with torch.no_grad():
        for x, t in loader:
            p.extend(model(x.to(device)).argmax(dim=1).cpu().tolist())
            y.extend(t.tolist())
    return y, p


def fit(model, train, val, device, epochs=8, lr=0.002, patience=2):
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    best, stale, state, history = -1.0, 0, None, []
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(loss.item())
        yt, yp = evaluate(model, val, device)
        score = f1_score(yt, yp, average="macro", zero_division=0)
        history.append({"epoch": ep, "loss": float(np.mean(losses)), "val_macro_f1": score})
        print(f"  epoch {ep:02d} loss={np.mean(losses):.3f} val_macro_f1={score:.3f}", flush=True)
        if score > best + 1e-4:
            best, stale = score, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(state)  # restore best-validation epoch
    return history


def run_experiment(cls, sets, vocab_size, n_labels, seed, args, device):
    set_seed(seed)
    model = cls(vocab_size, n_labels)
    gen = torch.Generator().manual_seed(seed)
    tl = DataLoader(sets["train"], batch_size=args.batch_size, shuffle=True, generator=gen)
    vl = DataLoader(sets["val"], batch_size=args.batch_size)
    hist = fit(model, tl, vl, device, args.epochs)
    y, p = evaluate(model, DataLoader(sets["test"], batch_size=args.batch_size), device)
    return model, hist, y, p


def bootstrap_ci(y, p, groups, n=500, seed=SEED):
    """95% CI for macro-F1, resampling whole question groups (answers to one prompt are not independent)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    by_group = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    keys = list(by_group)
    vals = []
    for _ in range(n):
        idx = np.concatenate([by_group[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        vals.append(f1_score(y[idx], p[idx], average="macro", zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------- baselines
def baselines(tr, va, te, labels, ablate_output=True):
    rows = []
    n = len(labels)
    majority = tr.label.value_counts().idxmax()
    pred = [majority] * len(te)
    rows.append({"baseline": "chance (uniform)", "features": "-", "accuracy": 1 / n, "macro_f1": np.nan})
    rows.append({"baseline": f"majority class ({majority})", "features": "-",
                 "accuracy": accuracy_score(te.label, pred),
                 "macro_f1": f1_score(te.label, pred, average="macro", zero_division=0)})
    conds = [("input", False), ("output", False), ("both", False)] + ([("output", True)] if ablate_output else [])
    for mode, ablate in conds:
        prep = strip_format if ablate else (lambda s: s)
        def txt(f):
            if mode == "input":
                return [prep(s) for s in f.prompt]
            if mode == "output":
                return [prep(s) for s in f.response]
            return [prep(p) + " <SEP> " + prep(r) for p, r in zip(f.prompt, f.response)]
        vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2),
                              min_df=2, max_features=100000, sublinear_tf=True)
        xtr = vec.fit_transform(txt(tr))  # fit on train only
        clf = LogisticRegression(max_iter=1000).fit(xtr, tr.label)
        p = clf.predict(vec.transform(txt(te)))
        rows.append({"baseline": "TF-IDF (1-2gram) + LogReg", "features": mode + ("_noformat" if ablate else ""),
                     "accuracy": accuracy_score(te.label, p),
                     "macro_f1": f1_score(te.label, p, average="macro", zero_division=0)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- RQ4
AI_PHRASES = re.compile(r"as an ai|language model|i'm sorry|i am sorry|i apologi[sz]e|i cannot|i can't", re.I)
FEATURES = ["word_count", "lexical_diversity", "ttr_first100", "bullet_lines", "code_fences",
            "newline_count", "avg_token_len", "ai_disclaimer_phrases"]


def response_features(frame):
    toks = frame.response.map(tokenize)
    words = toks.map(lambda t: [w for w in t if w != "<nl>"])
    d = pd.DataFrame(index=frame.index)
    d["word_count"] = words.map(len)
    d["lexical_diversity"] = words.map(lambda w: len(set(w)) / max(1, len(w)))
    d["ttr_first100"] = words.map(lambda w: len(set(w[:100])) / max(1, len(w[:100])))  # length-controlled
    d["bullet_lines"] = frame.response.map(lambda x: sum(bool(re.match(r"^\s*([-*]|\d+[.)])\s+", l)) for l in x.splitlines()))
    d["code_fences"] = frame.response.str.count("```")
    d["newline_count"] = frame.response.str.count("\n")
    d["avg_token_len"] = words.map(lambda w: np.mean([len(t) for t in w]) if w else 0.0)
    d["ai_disclaimer_phrases"] = frame.response.map(lambda x: len(AI_PHRASES.findall(x)))
    return d


def rq4(df, tr, te, labels, out):
    feats = response_features(df)
    feats["label"] = df.label.values
    feats.groupby("label")[FEATURES].agg(["mean", "median"]).round(3).to_csv(out / "rq4_features.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col, title in zip(axes.ravel(), ["word_count", "ttr_first100", "bullet_lines", "newline_count"],
                              ["Response length (tokens)", "Lexical diversity (first 100 tokens)",
                               "List lines per response", "Newlines per response"]):
        feats.boxplot(column=col, by="label", ax=ax, showfliers=False, rot=25)
        ax.set_title(title); ax.set_xlabel("")
    plt.suptitle("Response characteristics by model family")
    fig.tight_layout(); fig.savefig(out / "rq4_features.png", dpi=160); plt.close(fig)
    # Keep the older single-panel name for compatibility with the report generator
    fig, ax = plt.subplots(figsize=(8, 4))
    feats.boxplot(column="word_count", by="label", ax=ax, showfliers=False)
    plt.suptitle(""); ax.set_title("Response length by model family"); ax.set_ylabel("Tokens")
    fig.tight_layout(); fig.savefig(out / "rq4_response_lengths.png", dpi=160); plt.close(fig)

    # How far do ONLY these interpretable features get us? (train on train, score on test)
    ftr, fte = response_features(tr), response_features(te)
    clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=200).fit(ftr, tr.label)
    pred = clf.predict(fte)
    pd.DataFrame([{"model": "GradientBoosting on 8 handcrafted features",
                   "accuracy": accuracy_score(te.label, pred),
                   "macro_f1": f1_score(te.label, pred, average="macro", zero_division=0)}]
                 ).to_csv(out / "rq4_feature_classifier.csv", index=False)
    imp = permutation_importance(clf, fte, te.label, scoring="f1_macro", n_repeats=5, random_state=SEED)
    pd.DataFrame({"feature": FEATURES, "macro_f1_drop": imp.importances_mean, "std": imp.importances_std}
                 ).sort_values("macro_f1_drop", ascending=False).to_csv(out / "rq4_feature_importance.csv", index=False)


# --------------------------------------------------------------------------- RQ3 (optional)
DOMAINS = [
    ("coding", re.compile(r"```|\b(python|javascript|java|c\+\+|c#|sql|html|css|react|function|code|script|regex|algorithm|compile|bug|api|array|class)\b|def \w+\(|#include", re.I)),
    ("math", re.compile(r"\b(solve|equation|integral|derivative|probability|calculate|prove|theorem|how many|math)\b|\d+\s*[+\-*/^=]\s*\d+", re.I)),
    ("writing", re.compile(r"\b(write|essay|story|poem|email|letter|article|blog|song|lyrics|rewrite|paragraph|summari[sz]e)\w*", re.I)),
]


def domain_of(prompt):
    for name, pat in DOMAINS:  # first match wins; heuristic keyword labels, not ground truth
        if pat.search(prompt):
            return name
    return "general"


def rq3(df, te, test_preds, labels, args, device, out):
    lookup = {v: i for i, v in enumerate(labels)}
    df = df.assign(domain=df.prompt.map(domain_of))
    te = te.assign(domain=te.prompt.map(domain_of))
    rows = []
    for dom in sorted(df.domain.unique()):
        held, rest = df[df.domain == dom], df[df.domain != dom]
        if len(held) < 150 or held.label.nunique() < len(labels) or rest.label.nunique() < len(labels):
            print(f"RQ3: skipping domain '{dom}' (too small)"); continue
        r_tr, r_va = group_split(rest.reset_index(drop=True), 0.1)
        toks = {s: build_tokens(f, "output", args.seq_len) for s, f in (("train", r_tr), ("val", r_va), ("test", held))}
        vocab = vocab_fit(toks["train"])
        sets = {s: TextSet(toks[s], [lookup[l] for l in f.label], vocab, args.seq_len)
                for s, f in (("train", r_tr), ("val", r_va), ("test", held))}
        for kind, cls in (("CNN", TextCNN), ("BiLSTM", TextLSTM)):
            print(f"RQ3 {kind}: hold out '{dom}'", flush=True)
            _, _, y, p = run_experiment(cls, sets, len(vocab), len(labels), args.seeds[0], args, device)
            ty, tp = test_preds[(kind, "output")]
            mask = (te.domain == dom).values
            in_f1 = np.nan
            if mask.sum() >= 30:
                yy, pp = np.asarray(ty)[mask], np.asarray(tp)[mask]
                in_f1 = f1_score(yy, pp, labels=sorted(set(yy)), average="macro", zero_division=0)
            rows.append({"model": kind, "held_out_domain": dom, "n_held_out": len(held),
                         "ood_accuracy": accuracy_score(y, p),
                         "ood_macro_f1": f1_score(y, p, average="macro", zero_division=0),
                         "in_domain_macro_f1": in_f1})
    pd.DataFrame(rows).to_csv(out / "rq3.csv", index=False)


# --------------------------------------------------------------------------- plots
def plot_matrix(y, p, labels, path):
    cm = confusion_matrix(y, p, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right"); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-per-family", type=int, default=1600)
    ap.add_argument("--min-per-family", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=180)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42], help="model-init seeds (split stays fixed)")
    ap.add_argument("--rq3", action="store_true", help="run leave-one-domain-out generalisation (extra credit)")
    ap.add_argument("--skip-ablation", action="store_true", help="skip the formatting-removed condition")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(SEED)
    df, raw_counts = read_data(args.data, args.max_per_family, args.min_per_family)
    tr, va, te, labels = make_splits(df)
    lookup = {v: i for i, v in enumerate(labels)}
    print("device:", device, "| counts:", df.label.value_counts().to_dict(), "| splits:", len(tr), len(va), len(te), flush=True)

    raw_counts.rename("n").to_csv(out / "dataset_counts_before_cap.csv", header=True)
    df.groupby("label").size().rename("n").to_csv(out / "dataset_counts.csv")
    pd.DataFrame([{"split": s, "family": lab, "count": int((f.label == lab).sum())}
                  for s, f in (("train", tr), ("val", va), ("test", te)) for lab in labels]
                 ).to_csv(out / "split_counts.csv", index=False)
    json.dump({"args": vars(args), "device": device, "torch": torch.__version__, "python": sys.version.split()[0],
               "labels": labels, "n_train": len(tr), "n_val": len(va), "n_test": len(te)},
              open(out / "run_config.json", "w"), indent=2)

    rq4(df, tr, te, labels, out)
    if not args.skip_baselines:
        baselines(tr, va, te, labels, not args.skip_ablation).to_csv(out / "baselines.csv", index=False)

    conditions = [("input", "input", False), ("output", "output", False), ("both", "both", False)]
    if not args.skip_ablation:
        conditions.append(("output_noformat", "output", True))
    cache = {}
    for key, mode, ablate in conditions:
        toks = {s: build_tokens(f, mode, args.seq_len, ablate) for s, f in (("train", tr), ("val", va), ("test", te))}
        vocab = vocab_fit(toks["train"])  # train-only vocabulary
        cache[key] = (vocab, {s: TextSet(toks[s], [lookup[l] for l in f.label], vocab, args.seq_len)
                              for s, f in (("train", tr), ("val", va), ("test", te))})

    metrics, pred_rows, test_preds = [], [], {}
    for seed in args.seeds:
        first = seed == args.seeds[0]
        for key, _, _ in conditions:
            vocab, sets = cache[key]
            for kind, cls in (("CNN", TextCNN), ("BiLSTM", TextLSTM)):
                print(f"\nTraining {kind} | {key} | seed {seed}", flush=True)
                model, hist, y, p = run_experiment(cls, sets, len(vocab), len(labels), seed, args, device)
                row = {"model": kind, "features": key, "seed": seed, "accuracy": accuracy_score(y, p),
                       "macro_f1": f1_score(y, p, average="macro", zero_division=0),
                       "weighted_f1": f1_score(y, p, average="weighted", zero_division=0), "n_test": len(y)}
                if first:
                    row["macro_f1_ci_low"], row["macro_f1_ci_high"] = bootstrap_ci(y, p, te.group.tolist())
                    test_preds[(kind, key)] = (y, p)
                    pd.DataFrame(hist).to_csv(out / f"training_{kind}_{key}.csv", index=False)
                    plot_matrix(y, p, labels, out / f"confusion_{kind}_{key}.png")
                    (out / f"classification_{kind}_{key}.txt").write_text(classification_report(
                        y, p, target_names=labels, labels=range(len(labels)), zero_division=0))
                    pred_rows.extend({"group": g, "model": kind, "features": key, "true_family": labels[t],
                                      "predicted_family": labels[z]} for g, t, z in zip(te.group, y, p))
                    torch.save({"model_state": model.state_dict(), "vocab": vocab, "labels": labels,
                                "architecture": kind, "features": key, "seq_len": args.seq_len},
                               out / f"model_{kind}_{key}.pt")
                metrics.append(row)
                pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)

    m = pd.DataFrame(metrics)
    summary = m.groupby(["model", "features"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"), n_seeds=("seed", "nunique")).reset_index()
    summary.to_csv(out / "metrics_summary.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "predictions.csv", index=False)
    if args.rq3:
        rq3(df, te, test_preds, labels, args, device, out)
    print("\nFINAL TEST SCORES\n", summary.to_string(index=False))
    print("Saved to", out.resolve())


if __name__ == "__main__":
    main()
