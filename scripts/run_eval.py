"""Evaluation harness: run tickets through multiple model configurations and produce a comparison report."""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.types import Command

from src.checkpointer import get_checkpointer
from src.graph import build_graph
from src.models import PRESET_CONFIGS

load_dotenv()

EVAL_TICKETS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_tickets.json"

INITIAL_STATE = {
    "messages": [],
    "ticket_text": "",
    "category": "",
    "confidence": 0.0,
    "reasoning": "",
    "response": "",
    "retrieved_docs": [],
    "quality_approved": False,
    "quality_feedback": "",
}

COST_PER_1M = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "deepseek-chat": {"input": 0.27, "output": 1.10},
}


class TokenCounter(BaseCallbackHandler):
    """Callback handler that accumulates token counts across LLM calls."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        for generation_list in response.generations:
            for gen in generation_list:
                info = getattr(gen, "generation_info", None) or {}
                usage = info.get("usage", {})
                self.input_tokens += usage.get("input_tokens", 0) or usage.get(
                    "prompt_tokens", 0
                )
                self.output_tokens += usage.get("output_tokens", 0) or usage.get(
                    "completion_tokens", 0
                )


def load_eval_tickets(path=EVAL_TICKETS_PATH):
    """Load evaluation tickets from JSON file."""
    with open(path) as f:
        return json.load(f)


def run_ticket(app, ticket, thread_id):
    """Run a single ticket through the graph, auto-resuming any interrupts."""
    config = {"configurable": {"thread_id": thread_id}}
    state = {**INITIAL_STATE, "ticket_text": ticket["ticket_text"]}

    start = time.perf_counter()
    result = app.invoke(state, config=config)

    graph_state = app.get_state(config)
    while graph_state.tasks and any(
        hasattr(t, "interrupts") and t.interrupts for t in graph_state.tasks
    ):
        interrupt_info = graph_state.tasks[0].interrupts[0].value
        if "current_category" in interrupt_info:
            resume_value = interrupt_info["current_category"]
        else:
            resume_value = "approve"
        result = app.invoke(Command(resume=resume_value), config=config)
        graph_state = app.get_state(config)

    latency = time.perf_counter() - start
    return {**result, "_latency": latency}


def run_config_eval(config_name, model_config, tickets):
    """Run all tickets through a single model configuration."""
    print(f"\n--- Evaluating: {config_name} ---")
    checkpointer = get_checkpointer()
    app = build_graph(model_config=model_config, checkpointer=checkpointer)

    results = []
    for i, ticket in enumerate(tickets):
        thread_id = f"eval-{config_name}-{ticket['id']}"
        print(f"  [{i + 1}/{len(tickets)}] {ticket['id']}: {ticket['ticket_text'][:60]}...")
        try:
            result = run_ticket(app, ticket, thread_id)
            results.append(
                {
                    "ticket_id": ticket["id"],
                    "expected": ticket["expected_category"],
                    "predicted": result["category"],
                    "correct": result["category"] == ticket["expected_category"],
                    "confidence": result["confidence"],
                    "quality_approved": result["quality_approved"],
                    "latency": result["_latency"],
                    "response_length": len(result["response"]),
                }
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append(
                {
                    "ticket_id": ticket["id"],
                    "expected": ticket["expected_category"],
                    "predicted": "ERROR",
                    "correct": False,
                    "confidence": 0.0,
                    "quality_approved": False,
                    "latency": 0.0,
                    "response_length": 0,
                    "error": str(e),
                }
            )
    return results


def compute_metrics(results):
    """Compute aggregate metrics from a list of ticket results."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    quality_passed = sum(1 for r in results if r["quality_approved"])
    latencies = [r["latency"] for r in results if r["latency"] > 0]

    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "quality_pass_rate": quality_passed / total if total else 0,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "avg_confidence": (
            sum(r["confidence"] for r in results) / total if total else 0
        ),
    }

    for category in ("billing", "technical", "general"):
        cat_results = [r for r in results if r["expected"] == category]
        cat_total = len(cat_results)
        cat_correct = sum(1 for r in cat_results if r["correct"])
        cat_latencies = [r["latency"] for r in cat_results if r["latency"] > 0]
        metrics[f"{category}_accuracy"] = cat_correct / cat_total if cat_total else 0
        metrics[f"{category}_correct"] = cat_correct
        metrics[f"{category}_total"] = cat_total
        metrics[f"{category}_avg_latency"] = (
            sum(cat_latencies) / len(cat_latencies) if cat_latencies else 0
        )
        metrics[f"{category}_avg_confidence"] = (
            sum(r["confidence"] for r in cat_results) / cat_total if cat_total else 0
        )

    return metrics


def generate_report(all_config_results, output_path):
    """Generate a markdown comparison report from evaluation results."""
    lines = []
    lines.append("# Evaluation Report")
    lines.append("")

    config_names = list(all_config_results.keys())
    all_metrics = {
        name: compute_metrics(results) for name, results in all_config_results.items()
    }

    # Summary table
    lines.append("## Summary")
    lines.append("")
    header = "| Metric | " + " | ".join(config_names) + " |"
    sep = "|--------|" + "|".join(["--------"] * len(config_names)) + "|"
    lines.append(header)
    lines.append(sep)

    for label, key, fmt in [
        ("Accuracy", "accuracy", lambda v: f"{v:.0%} ({all_metrics[n]['correct']}/{all_metrics[n]['total']})"),
        ("Quality Pass Rate", "quality_pass_rate", lambda v: f"{v:.0%}"),
        ("Avg Latency", "avg_latency", lambda v: f"{v:.1f}s"),
        ("Avg Confidence", "avg_confidence", lambda v: f"{v:.2f}"),
    ]:
        row = f"| {label} |"
        for n in config_names:
            val = all_metrics[n][key]
            row += f" {fmt(val)} |"
        lines.append(row)
    lines.append("")

    # Per-category breakdown
    lines.append("## Per-Category Breakdown")
    lines.append("")
    for category in ("billing", "technical", "general"):
        lines.append(f"### {category.title()}")
        lines.append("")
        header = "| Metric | " + " | ".join(config_names) + " |"
        lines.append(header)
        lines.append(sep)
        for label, key, fmt in [
            ("Accuracy", f"{category}_accuracy", lambda v: f"{v:.0%}"),
            ("Correct", f"{category}_correct", lambda v: f"{int(v)}/{all_metrics[n][f'{category}_total']}"),
            ("Avg Latency", f"{category}_avg_latency", lambda v: f"{v:.1f}s"),
            ("Avg Confidence", f"{category}_avg_confidence", lambda v: f"{v:.2f}"),
        ]:
            row = f"| {label} |"
            for n in config_names:
                val = all_metrics[n][key]
                row += f" {fmt(val)} |"
            lines.append(row)
        lines.append("")

    # Detailed per-ticket results
    lines.append("## Detailed Results")
    lines.append("")
    header = "| Ticket | Expected | " + " | ".join(config_names) + " |"
    sep_detail = "|--------|----------|" + "|".join(["--------"] * len(config_names)) + "|"
    lines.append(header)
    lines.append(sep_detail)

    first_config = config_names[0]
    for ticket_result in all_config_results[first_config]:
        tid = ticket_result["ticket_id"]
        expected = ticket_result["expected"]
        row = f"| {tid} | {expected} |"
        for n in config_names:
            match = next((r for r in all_config_results[n] if r["ticket_id"] == tid), None)
            if match:
                mark = "+" if match["correct"] else "-"
                row += f" {mark} {match['predicted']} ({match['confidence']:.2f}) |"
            else:
                row += " N/A |"
        lines.append(row)
    lines.append("")

    report = "\n".join(lines)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    print(f"\nReport written to {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(PRESET_CONFIGS.keys()),
        choices=list(PRESET_CONFIGS.keys()),
        help="Model config presets to evaluate",
    )
    parser.add_argument(
        "--output",
        default="reports/eval_report.md",
        help="Output report path",
    )
    args = parser.parse_args()

    tickets = load_eval_tickets()
    print(f"Loaded {len(tickets)} evaluation tickets")

    all_results = {}
    for config_name in args.configs:
        model_config = PRESET_CONFIGS[config_name]
        results = run_config_eval(config_name, model_config, tickets)
        all_results[config_name] = results

    generate_report(all_results, args.output)


if __name__ == "__main__":
    main()
