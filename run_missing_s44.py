"""Finish seed 44: run only the conditions missing from results_s44/metrics.csv (same split, vocab and settings as project.py)."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import project as P

out = Path("results_s44"); mp = out / "metrics.csv"
done = pd.read_csv(mp); seed = 44
df, _ = P.read_data("data/chatbot_arena.parquet")
tr, va, te, labels = P.make_splits(df)
lookup = {v: i for i, v in enumerate(labels)}
args = argparse.Namespace(seq_len=180, batch_size=64, epochs=8, seeds=[seed])
conds = [("input", "input", False), ("output", "output", False), ("both", "both", False), ("output_noformat", "output", True)]
rows = done.to_dict("records")
for key, mode, ablate in conds:
    todo = [(k, c) for k, c in (("CNN", P.TextCNN), ("BiLSTM", P.TextLSTM))
            if not ((done.model == k) & (done.features == key) & (done.seed == seed)).any()]
    if not todo: continue
    toks = {s: P.build_tokens(f, mode, args.seq_len, ablate) for s, f in (("train", tr), ("val", va), ("test", te))}
    vocab = P.vocab_fit(toks["train"])
    sets = {s: P.TextSet(toks[s], [lookup[l] for l in f.label], vocab, args.seq_len) for s, f in (("train", tr), ("val", va), ("test", te))}
    for kind, cls in todo:
        print(f"\nTraining {kind} | {key} | seed {seed}", flush=True)
        _, _, y, p = P.run_experiment(cls, sets, len(vocab), len(labels), seed, args, "cpu")
        rows.append({"model": kind, "features": key, "seed": seed, "accuracy": accuracy_score(y, p),
                     "macro_f1": f1_score(y, p, average="macro", zero_division=0),
                     "weighted_f1": f1_score(y, p, average="weighted", zero_division=0), "n_test": len(y)})
        pd.DataFrame(rows).to_csv(mp, index=False)
print("SEED44 DONE", flush=True)
