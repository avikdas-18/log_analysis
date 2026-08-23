"""Command-line entry point for local CAN log RAG queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from can_log_analyzer import CanLogDataset, result_summary
from local_llm import LocalGGUFLLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Query CAN signal CSV logs locally.")
    parser.add_argument("csv", type=Path, help="CAN log CSV path")
    parser.add_argument("question", help="Example: 'pack_current exceeds 80A'")
    parser.add_argument("--model", required=True, help="Path to a local GGUF instruct model")
    parser.add_argument("--context-size", type=int, default=4096, help="LLM context size")
    parser.add_argument("--gpu-layers", type=int, default=0, help="GPU-offloaded LLM layers; 0 uses CPU")
    parser.add_argument("--output", type=Path, help="Optional CSV path for matching rows")
    args = parser.parse_args()

    dataset = CanLogDataset.from_csv(args.csv, args.csv.name)
    llm = LocalGGUFLLM(args.model, context_size=args.context_size, gpu_layers=args.gpu_layers)
    result = dataset.execute(llm.plan(args.question, dataset))
    print(llm.answer(args.question, result))
    print(result_summary(result))
    print(f"Resolved signal: {result['matched_signal']}")
    print(f"Native cycle time: {result['cycle_time']}")
    print(json.dumps(result["matching_periods"], indent=2))
    if args.output:
        pd.DataFrame(result["rows"]).to_csv(args.output, index=False)
        print(f"Matching rows written to {args.output}")


if __name__ == "__main__":
    main()
