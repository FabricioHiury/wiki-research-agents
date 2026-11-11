"""CLI entrypoint for running the Strands Agents pipeline: python -m strands_impl "topic"."""

from __future__ import annotations

import sys

from shared.config import get_config
from shared.formatting import print_banner, print_final_summary, print_metrics
from strands_impl.pipeline import run_strands_pipeline


def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else "Artificial Intelligence"
    config = get_config()

    print_banner("Strands Agents", topic)

    result = run_strands_pipeline(topic, config=config, check_server=True)

    if not result["success"]:
        print(f"\n[ERROR] Pipeline aborted: {result.get('error')}\n", file=sys.stderr)
        sys.exit(1)

    print_final_summary(result["final_summary"], "Strands Agents")
    print_metrics(
        elapsed_seconds=result["elapsed_seconds"],
        llm_calls=result["llm_calls"],
        tool_calls=result["tool_calls"],
    )


if __name__ == "__main__":
    main()

