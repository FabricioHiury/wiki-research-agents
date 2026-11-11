"""Side-by-side benchmark comparison between LangGraph and Strands Agents.

Runs the identical multi-agent research pipeline across both frameworks
using the same local Ollama model and identical system prompts.
"""

from __future__ import annotations

import sys
from typing import Any

from langgraph_impl.graph import run_langgraph_pipeline
from shared.config import get_config
from shared.formatting import (
    Colors,
    print_comparison_table,
    print_final_summary,
)
from strands_impl.pipeline import run_strands_pipeline


def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else "Artificial Intelligence"
    config = get_config()

    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*78}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}  MULTI-AGENT FRAMEWORK COMPARISON BENCHMARK{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}  Topic:      {topic}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}  Model:      {config.model_name}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}  Ollama:     {config.ollama_host}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*78}{Colors.END}\n")

    results: list[dict[str, Any]] = []

    # 1. Run LangGraph Pipeline
    print(f"{Colors.BOLD}{Colors.CYAN}>>> Executing LangGraph Implementation...{Colors.END}\n")
    lg_res = run_langgraph_pipeline(topic, config=config, check_server=True)
    results.append(lg_res)

    print()

    # 2. Run Strands Agents Pipeline
    print(f"{Colors.BOLD}{Colors.CYAN}>>> Executing Strands Agents Implementation...{Colors.END}\n")
    st_res = run_strands_pipeline(topic, config=config, check_server=True)
    results.append(st_res)

    # 3. Print Final Summaries side-by-side
    if lg_res.get("success"):
        print_final_summary(lg_res["final_summary"], "LangGraph")
    else:
        print(f"\n{Colors.RED}[LangGraph Failed]{Colors.END} {lg_res.get('error')}")

    if st_res.get("success"):
        print_final_summary(st_res["final_summary"], "Strands Agents")
    else:
        print(f"\n{Colors.RED}[Strands Failed]{Colors.END} {st_res.get('error')}")

    # 4. Print Side-by-Side Comparison Table
    print_comparison_table(results)


if __name__ == "__main__":
    main()
