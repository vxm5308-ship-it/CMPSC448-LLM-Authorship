"""Run only RQ3 (leave-one-domain-out), re-using the saved seed-42 output-only checkpoints for the in-domain reference."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import project as P

out = Path("results")
df, _ = P.read_data("data/chatbot_arena.parquet")
tr, va, te, labels = P.make_splits(df)
lookup = {v: i for i, v in enumerate(labels)}
test_preds = {}
for kind, cls in (("CNN", P.TextCNN), ("BiLSTM", P.TextLSTM)):
    ck = torch.load(out / f"model_{kind}_output.pt", weights_only=False)
    assert ck["labels"] == labels
    m = cls(len(ck["vocab"]), len(labels)); m.load_state_dict(ck["model_state"])
    toks = P.build_tokens(te, "output", ck["seq_len"])
    ds = P.TextSet(toks, [lookup[l] for l in te.label], ck["vocab"], ck["seq_len"])
    test_preds[(kind, "output")] = P.evaluate(m, DataLoader(ds, batch_size=64), "cpu")
    print(kind, "reloaded test acc", sum(a == b for a, b in zip(*test_preds[(kind, "output")])) / len(te), flush=True)
args = argparse.Namespace(seq_len=180, batch_size=64, epochs=8, seeds=[42])
P.rq3(df, te, test_preds, labels, args, "cpu", out)
print("RQ3 done", flush=True)
