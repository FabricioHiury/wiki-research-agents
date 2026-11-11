"""Terminal output formatting, banners, and performance metrics display."""

from __future__ import annotations

import sys
from typing import Any


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_banner(framework: str, topic: str) -> None:
    """Print an eye-catching CLI execution banner."""
    bar = "=" * 70
    print(f"\n{Colors.BOLD}{Colors.CYAN}{bar}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}  MULTI-AGENT RESEARCH PIPELINE: {framework.upper()}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}  Topic: {topic}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{bar}{Colors.END}\n")


def print_agent_step(agent_name: str, status: str, details: str = "") -> None:
    """Log an agent pipeline step to stdout."""
    prefix_colors = {
        "Researcher": Colors.YELLOW,
        "Analyst": Colors.CYAN,
        "Summarizer": Colors.GREEN,
        "System": Colors.BLUE,
    }
    color = prefix_colors.get(agent_name, Colors.BOLD)
    print(f"{color}[{agent_name}]{Colors.END} {status}")
    if details:
        preview = details.strip()
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"  {Colors.BOLD}↳{Colors.END} {preview}\n")


def print_final_summary(summary: str, framework: str) -> None:
    """Print final synthesized summary."""
    divider = "-" * 70
    print(f"\n{Colors.BOLD}{Colors.GREEN}{divider}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}  FINAL EXECUTIVE SUMMARY ({framework}){Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}{divider}{Colors.END}\n")
    print(summary.strip())
    print(f"\n{Colors.BOLD}{Colors.GREEN}{divider}{Colors.END}\n")


def print_metrics(elapsed_seconds: float, llm_calls: int, tool_calls: int = 0) -> None:
    """Print execution metrics."""
    print(f"{Colors.BOLD}Execution Metrics:{Colors.END}")
    print(f"  • Total Time:       {Colors.BOLD}{elapsed_seconds:.2f}s{Colors.END}")
    print(f"  • LLM Invocations:  {Colors.BOLD}{llm_calls}{Colors.END}")
    if tool_calls:
        print(f"  • Tool Invocations: {Colors.BOLD}{tool_calls}{Colors.END}")
    print()


def print_comparison_table(
    results: list[dict[str, Any]],
) -> None:
    """Print side-by-side comparison table between frameworks."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*78}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}  SIDE-BY-SIDE FRAMEWORK BENCHMARK RESULTS{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*78}{Colors.END}")
    print(
        f"{Colors.BOLD}{'Framework':<20} | {'Status':<10} | {'Elapsed (s)':<12} | "
        f"{'LLM Calls':<10} | {'Tool Calls':<10}{Colors.END}"
    )
    print(f"{'-'*78}")
    for res in results:
        status_col = (
            f"{Colors.GREEN}SUCCESS{Colors.END}"
            if res.get("success")
            else f"{Colors.RED}FAILED{Colors.END}"
        )
        elapsed = f"{res.get('elapsed_seconds', 0.0):.2f}"
        llm_calls = str(res.get("llm_calls", 0))
        tool_calls = str(res.get("tool_calls", 0))
        print(
            f"{res['framework']:<20} | {status_col:<19} | {elapsed:<12} | "
            f"{llm_calls:<10} | {tool_calls:<10}"
        )
    print(f"{Colors.BOLD}{'='*78}{Colors.END}\n")
