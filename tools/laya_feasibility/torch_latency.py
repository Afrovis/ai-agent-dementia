"""Time a Laya checkpoint under torch: one question, one 32-question decision, one 32-option choice.

python torch_latency.py --device mps
python torch_latency.py --device mps --subfolder multilingual
"""

import argparse
import json
import os
import platform
import time

os.environ.setdefault("USE_TF", "0")

import laya
import torch

from probe import (
    STATE,
    action_choice_32,
    action_question,
    all_action_questions,
    peak_rss_mb,
    strategy_choice,
    time_it,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    agent = laya.load("convaiinnovations/laya", device=args.device, subfolder=args.subfolder)
    load_s = time.perf_counter() - t0
    if args.fp16:
        # laya pins MPS to fp32; the inputs are integer ids, so halving the weights is enough.
        # act_head is fed `pooled.float()`, so it has to stay fp32.
        agent.model.half()
        agent.model.act_head.float()
        agent.dtype = torch.float16

    one = {"q": action_question("goal", "restroom")}
    decision = all_action_questions()
    tokens = agent.predict(STATE, one)["usage"]["input_tokens"]

    result = {
        "checkpoint": args.subfolder or "english",
        "device": str(agent.device),
        "dtype": str(agent.dtype),
        "torch": torch.__version__,
        "laya": getattr(laya, "__version__", "?"),
        "machine": platform.machine(),
        "load_s": round(load_s, 1),
        "tokens_per_question": tokens,
        "single_question": time_it(lambda: agent.predict(STATE, one), args.reps),
        "decision_32_bool": time_it(lambda: agent.predict(STATE, decision), max(5, args.reps // 3)),
        "choice_8_strategies": time_it(
            lambda: agent.predict(STATE, {"q": strategy_choice()}), args.reps
        ),
        "choice_32_actions": time_it(
            lambda: agent.predict(STATE, {"q": action_choice_32()}), args.reps
        ),
        "peak_rss_mb": peak_rss_mb(),
    }
    if agent.device.type == "mps":
        result["mps_driver_allocated_mb"] = round(torch.mps.driver_allocated_memory() / 2**20, 1)

    # Zero-shot answers, for the record only: accuracy is not the Phase 0 gate.
    answers = agent.predict(STATE, decision)["answers"]
    result["zero_shot_bool"] = {k: v["noul"] for k, v in answers.items()}
    result["zero_shot_strategy"] = agent.predict(STATE, {"q": strategy_choice()})["answers"]["q"]

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
