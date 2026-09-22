"""Compare FluidInference/laya-coreml against the torch multilingual checkpoint it came from,
and time it on the ANE.

The Core ML buckets are converted from `convaiinnovations/laya` `multilingual/` at revision
1c5edc17, so that is the torch reference here, not the English checkpoint. Inputs are built
with laya's own `build_sequence`, so both sides see identical token ids and marker positions.

    python coreml_parity.py --bucket 256
"""

import argparse
import json
import os
import time

os.environ.setdefault("USE_TF", "0")

import coremltools as ct
import laya
import numpy as np
import torch
from huggingface_hub import snapshot_download
from laya.common import QTYPES, build_sequence

from probe import (
    STATE,
    action_choice_32,
    all_action_questions,
    peak_rss_mb,
    strategy_choice,
    time_it,
)

REPO = "FluidInference/laya-coreml"
MAX_OPTIONS = 32
UNITS = {
    "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
}


def coreml_inputs(agent, q_internal: dict, length: int) -> dict:
    head_max_len = agent.cfg.get("head_max_len", 256)
    seq, markers = build_sequence(agent.tok, STATE, q_internal, length, head_max_len)
    ids = np.full((1, length), agent.tok.pad_token_id, dtype=np.int32)
    att = np.zeros((1, length), dtype=np.int32)
    ids[0, : len(seq)] = seq
    att[0, : len(seq)] = 1
    marker_map = np.zeros((1, MAX_OPTIONS, length), dtype=np.float32)
    for j, pos in enumerate(markers):
        marker_map[0, j, pos] = 1.0
    qtype = np.zeros((1, 3), dtype=np.float32)
    qtype[0, QTYPES[q_internal["t"]]] = 1.0
    return (
        {
            "input_ids": ids,
            "attention_mask": att,
            "marker_map": marker_map,
            "question_type": qtype,
        },
        len(markers),
        len(seq),
    )


@torch.no_grad()
def torch_probs(agent, feed: dict, k: int) -> np.ndarray:
    """Untempered softmax over the k real options, matching the Core ML `probabilities` output."""
    n = int(feed["attention_mask"].sum())
    ids = torch.tensor(feed["input_ids"][:, :n], dtype=torch.long, device=agent.device)
    att = torch.ones_like(ids)
    pos = torch.tensor(np.argmax(feed["marker_map"][0, :k], axis=1)[None], device=agent.device)
    mask = torch.ones((1, k), dtype=torch.bool, device=agent.device)
    qt = torch.tensor([int(np.argmax(feed["question_type"][0]))], device=agent.device)
    logits, _ = agent.model(ids, att, pos, mask, qt)
    return torch.softmax(logits.float(), -1).cpu().numpy()[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", type=int, default=256)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "e8"])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name = f"laya_multilingual_{args.precision}_L{args.bucket}_options32.mlmodelc"
    root = snapshot_download(REPO, allow_patterns=[f"{name}/*", "config.json"])
    agent = laya.load("convaiinnovations/laya", device="cpu", subfolder="multilingual")

    questions = dict(all_action_questions())
    questions["choice:strategy"] = strategy_choice()
    questions["choice:action32"] = action_choice_32()
    feeds = {}
    for qid, qdef in questions.items():
        feed, k, n = coreml_inputs(agent, agent._to_internal(qdef), args.bucket)
        feeds[qid] = (feed, k, n)
    ref = {qid: torch_probs(agent, f, k) for qid, (f, k, n) in feeds.items()}

    result = {
        "model": name,
        "coremltools": ct.__version__,
        "tokens_per_question": sorted({n for _, _, n in feeds.values()}),
        "units": {},
    }
    for unit_name, unit in UNITS.items():
        t0 = time.perf_counter()
        model = ct.models.CompiledMLModel(os.path.join(root, name), compute_units=unit)
        load_s = time.perf_counter() - t0

        worst_abs, argmax_agree, noul_side_agree, n_noul = 0.0, 0, 0, 0
        per_q = {}
        for qid, (feed, k, _) in feeds.items():
            p_ml = model.predict(feed)["probabilities"][0, :k]
            p_t = ref[qid]
            diff = float(np.abs(p_ml - p_t).max())
            worst_abs = max(worst_abs, diff)
            argmax_agree += int(p_ml.argmax() == p_t.argmax())
            if k == 2:
                n_noul += 1
                noul_side_agree += int((p_ml[1] >= 0.5) == (p_t[1] >= 0.5))
            per_q[qid] = {"max_abs_diff": round(diff, 5), "coreml": p_ml.round(4).tolist()[:8]}

        one = feeds["goal:restroom"][0]
        bools = [f for qid, (f, k, _) in feeds.items() if k == 2]
        result["units"][unit_name] = {
            "load_s": round(load_s, 1),
            "parity": {
                "questions": len(feeds),
                "max_abs_prob_diff": round(worst_abs, 5),
                "argmax_agree": argmax_agree,
                "bool_side_agree": f"{noul_side_agree}/{n_noul}",
            },
            "single_question": time_it(lambda m=model, f=one: m.predict(f), args.reps),
            # Core ML buckets are batch 1, so a 32-question decision is 32 sequential calls.
            "decision_32_bool": time_it(
                lambda m=model, fs=bools: [m.predict(f) for f in fs], max(5, args.reps // 3)
            ),
            "choice_32_actions": time_it(
                lambda m=model, f=feeds["choice:action32"][0]: m.predict(f), args.reps
            ),
            "per_question": per_q if unit_name == "cpu_and_ne" else None,
        }
    result["peak_rss_mb"] = peak_rss_mb()

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
