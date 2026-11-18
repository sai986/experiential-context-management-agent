"""
Comparative analysis across all domains (incident / android / bgl / openstack).

Loads summary JSONs from:
  results/<domain>/processed/summary.json          (incident — real LLM)
  results/<domain>/simulated/processed/summary.json
  results/<domain>/real/processed/summary.json

Produces:
  results/comparative/
    01_success_by_condition.png  — grouped bar: success rate × condition × domain
    02_context_reduction.png     — grouped bar: avg active-context words × condition × domain
    03_learning_gain.png         — bar: lesson-aware A success − non-learning × domain
    04_transfer_effectiveness.png — bar: agent B success vs agent A vs non-learning
    05_family_heatmap.png        — heatmap: success by (domain × family), lesson-aware A
    06_real_vs_simulated.png     — scatter: simulated ceiling vs real ceiling per domain
    summary_table.txt            — ASCII table of all metrics
    combined_summary.json        — machine-readable combined results

Usage:
    python -m src.experiments.generate_comparative_analysis
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "comparative"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
COND_COLORS = {
    "full_context_test":          "#2196F3",   # blue
    "non_learning_manager_test":  "#FF5722",   # deep orange
    "lesson_aware_agent_a_test":  "#4CAF50",   # green
    "lesson_aware_agent_b_test":  "#9C27B0",   # purple
}
COND_LABELS = {
    "full_context_test":          "Full Context\n(ceiling)",
    "non_learning_manager_test":  "Non-Learning\n(baseline)",
    "lesson_aware_agent_a_test":  "Lesson-Aware A\n(learned)",
    "lesson_aware_agent_b_test":  "Agent B\n(transferred)",
}
CONDITIONS = list(COND_COLORS.keys())

DOMAIN_ORDER   = ["incident", "android", "bgl", "openstack"]
DOMAIN_LABELS  = {
    "incident":   "Synthetic\n(incident)",
    "android":    "Android\nLogcat",
    "bgl":        "BGL\nServer",
    "openstack":  "OpenStack\nCloud",
}
DOMAIN_COLORS  = {
    "incident":   "#607D8B",
    "android":    "#00BCD4",
    "bgl":        "#FF9800",
    "openstack":  "#E91E63",
}

FAMILIES = [
    "hard_constraint",
    "delayed_dependency",
    "dependency_chain",
    "false_distractor",
    "negation_critical",
    "synonym_match",
]
FAMILY_LABELS = {
    "hard_constraint":    "Hard\nConstraint",
    "delayed_dependency": "Delayed\nDependency",
    "dependency_chain":   "Dependency\nChain",
    "false_distractor":   "False\nDistractor",
    "negation_critical":  "Negation\nCritical",
    "synonym_match":      "Synonym\nMatch",
}


# ── data loading ──────────────────────────────────────────────────────────────

def _load_summary(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_all() -> dict[str, dict]:
    """Return {domain: {mode: summary_dict}}.  mode ∈ {real, simulated}."""
    data: dict[str, dict] = {}

    # incident uses results/incident/processed/summary.json (always real LLM)
    inc = _load_summary(RESULTS_DIR / "incident" / "processed" / "summary.json")
    if inc:
        data["incident"] = {"real": inc, "simulated": None}

    for domain in ("android", "bgl", "openstack"):
        domain_data: dict[str, Optional[dict]] = {}
        for mode in ("simulated", "real"):
            path = RESULTS_DIR / domain / mode / "processed" / "summary.json"
            domain_data[mode] = _load_summary(path)
        data[domain] = domain_data

    return data


def get_metric(summary: dict, condition: str, metric: str) -> Optional[float]:
    s = summary.get("summary", {})
    cond_data = s.get(condition)
    if cond_data is None:
        return None
    return cond_data.get(metric)


def get_family_sr(summary: dict, condition: str, family: str) -> Optional[float]:
    s = summary.get("summary", {})
    cond_data = s.get(condition, {})
    return cond_data.get("by_family", {}).get(family, {}).get("success_rate")


# ── plot helpers ──────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.0%}"


# ── Figure 1: Success rate by condition, grouped by domain ────────────────────

def plot_success_by_condition(all_data: dict[str, dict]) -> None:
    """4 condition groups, bars per domain, using 'real' mode where available."""
    fig, ax = plt.subplots(figsize=(12, 5))

    n_conditions = len(CONDITIONS)
    n_domains    = len(DOMAIN_ORDER)
    bar_w        = 0.17
    group_gap    = 0.3
    x_centers    = np.arange(n_conditions) * (n_domains * bar_w + group_gap)

    for di, domain in enumerate(DOMAIN_ORDER):
        modes = all_data.get(domain, {})
        summary = modes.get("real") or modes.get("simulated")
        if summary is None:
            continue
        offsets = (np.arange(n_conditions) * 0) + (di - (n_domains - 1) / 2) * bar_w
        values = [
            get_metric(summary, cond, "success_rate") or 0.0
            for cond in CONDITIONS
        ]
        bars = ax.bar(
            x_centers + offsets,
            values,
            bar_w * 0.9,
            label=DOMAIN_LABELS[domain].replace("\n", " "),
            color=DOMAIN_COLORS[domain],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if val > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{val:.0%}",
                    ha="center", va="bottom", fontsize=6.5, rotation=45,
                )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [COND_LABELS[c].replace("\n", " ") for c in CONDITIONS], fontsize=9
    )
    ax.set_ylabel("Success Rate", fontsize=11)
    ax.set_title(
        "Task Success Rate by Condition and Domain\n(Real LLM where available)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "01_success_by_condition.png")


# ── Figure 2: Average active context words ────────────────────────────────────

def plot_context_words(all_data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    n_conditions = len(CONDITIONS)
    n_domains    = len(DOMAIN_ORDER)
    bar_w        = 0.17
    group_gap    = 0.3
    x_centers    = np.arange(n_conditions) * (n_domains * bar_w + group_gap)

    for di, domain in enumerate(DOMAIN_ORDER):
        modes = all_data.get(domain, {})
        summary = modes.get("real") or modes.get("simulated")
        if summary is None:
            continue
        offsets = (np.arange(n_conditions) * 0) + (di - (n_domains - 1) / 2) * bar_w
        values = [
            get_metric(summary, cond, "avg_active_context_words") or 0.0
            for cond in CONDITIONS
        ]
        bars = ax.bar(
            x_centers + offsets,
            values,
            bar_w * 0.9,
            label=DOMAIN_LABELS[domain].replace("\n", " "),
            color=DOMAIN_COLORS[domain],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if val > 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.0f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=45,
                )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [COND_LABELS[c].replace("\n", " ") for c in CONDITIONS], fontsize=9
    )
    ax.set_ylabel("Avg Active Context Words", fontsize=11)
    ax.set_title(
        "Average Active Context Words by Condition and Domain",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "02_context_words.png")


# ── Figure 3: Learning gain (lesson A vs non-learning) ───────────────────────

def plot_learning_gain(all_data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    domains  = [d for d in DOMAIN_ORDER if d in all_data]
    x        = np.arange(len(domains))
    bar_w    = 0.35

    gains, baselines, ceilings = [], [], []
    for domain in domains:
        modes   = all_data[domain]
        summary = modes.get("real") or modes.get("simulated")
        if summary is None:
            gains.append(0.0)
            baselines.append(0.0)
            ceilings.append(0.0)
            continue
        base    = get_metric(summary, "non_learning_manager_test", "success_rate") or 0.0
        learned = get_metric(summary, "lesson_aware_agent_a_test", "success_rate") or 0.0
        ceil_   = get_metric(summary, "full_context_test",         "success_rate") or 0.0
        gains.append(learned - base)
        baselines.append(base)
        ceilings.append(ceil_)

    bars = ax.bar(
        x, gains,
        bar_w,
        color=[
            "#4CAF50" if g > 0 else "#F44336"
            for g in gains
        ],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
    )
    for bar, val in zip(bars, gains):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            (bar.get_height() if val >= 0 else bar.get_height()) + 0.01
            if val >= 0
            else val - 0.03,
            f"{val:+.0%}",
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=10, fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DOMAIN_LABELS[d].replace("\n", " ") for d in domains], fontsize=10
    )
    ax.set_ylabel("Success Rate Improvement\n(Lesson-Aware A − Non-Learning)", fontsize=10)
    ax.set_title(
        "Learning Gain from Context-Management Lessons",
        fontsize=12, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "03_learning_gain.png")


# ── Figure 4: Transfer effectiveness ─────────────────────────────────────────

def plot_transfer(all_data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    domains = [d for d in DOMAIN_ORDER if d in all_data]
    x = np.arange(len(domains))
    bar_w = 0.25

    for ci, (cond, label, color) in enumerate([
        ("non_learning_manager_test",  "Baseline (no learning)",   COND_COLORS["non_learning_manager_test"]),
        ("lesson_aware_agent_a_test",  "Agent A (trained)",        COND_COLORS["lesson_aware_agent_a_test"]),
        ("lesson_aware_agent_b_test",  "Agent B (transferred)",    COND_COLORS["lesson_aware_agent_b_test"]),
    ]):
        values = []
        for domain in domains:
            modes   = all_data[domain]
            summary = modes.get("real") or modes.get("simulated")
            val     = get_metric(summary, cond, "success_rate") if summary else 0.0
            values.append(val or 0.0)

        bars = ax.bar(
            x + (ci - 1) * bar_w, values, bar_w * 0.9,
            label=label, color=color, alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.0%}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DOMAIN_LABELS[d].replace("\n", " ") for d in domains], fontsize=10
    )
    ax.set_ylabel("Success Rate", fontsize=11)
    ax.set_title(
        "Cross-Agent Lesson Transfer Effectiveness",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "04_transfer_effectiveness.png")


# ── Figure 5: Per-family heatmap ──────────────────────────────────────────────

def plot_family_heatmap(all_data: dict[str, dict]) -> None:
    domains = [d for d in DOMAIN_ORDER if d in all_data]
    matrix  = np.full((len(FAMILIES), len(domains)), np.nan)

    for di, domain in enumerate(domains):
        modes   = all_data[domain]
        summary = modes.get("real") or modes.get("simulated")
        if summary is None:
            continue
        for fi, family in enumerate(FAMILIES):
            val = get_family_sr(summary, "lesson_aware_agent_a_test", family)
            if val is not None:
                matrix[fi, di] = val

    fig, ax = plt.subplots(figsize=(9, 5))
    masked = np.ma.masked_invalid(matrix)
    cmap   = plt.cm.RdYlGn.copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(
        [DOMAIN_LABELS[d].replace("\n", " ") for d in domains], fontsize=10
    )
    ax.set_yticks(range(len(FAMILIES)))
    ax.set_yticklabels(
        [FAMILY_LABELS[f].replace("\n", " ") for f in FAMILIES], fontsize=9
    )
    ax.set_title(
        "Per-Family Success Rate — Lesson-Aware Agent A",
        fontsize=12, fontweight="bold",
    )

    for fi in range(len(FAMILIES)):
        for di in range(len(domains)):
            val = matrix[fi, di]
            if not np.isnan(val):
                ax.text(
                    di, fi, f"{val:.0%}",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="black" if 0.3 < val < 0.8 else "white",
                )
            else:
                ax.text(di, fi, "N/A", ha="center", va="center", fontsize=9, color="grey")

    plt.colorbar(im, ax=ax, label="Success Rate", fraction=0.03, pad=0.03)
    _save(fig, "05_family_heatmap.png")


# ── Figure 6: Real vs simulated ceiling ───────────────────────────────────────

def plot_real_vs_simulated(all_data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for domain in DOMAIN_ORDER:
        modes = all_data.get(domain, {})
        sim   = modes.get("simulated")
        real  = modes.get("real")
        if sim is None and real is None:
            continue

        sim_sr  = get_metric(sim,  "full_context_test", "success_rate") if sim  else None
        real_sr = get_metric(real, "full_context_test", "success_rate") if real else None

        if sim_sr is not None and real_sr is not None:
            ax.scatter(
                sim_sr, real_sr,
                color=DOMAIN_COLORS[domain],
                s=180, zorder=5, label=DOMAIN_LABELS[domain].replace("\n", " "),
            )
            ax.annotate(
                DOMAIN_LABELS[domain].replace("\n", " "),
                (sim_sr, real_sr), textcoords="offset points",
                xytext=(8, 4), fontsize=9,
            )
        elif sim_sr is not None:
            ax.scatter(sim_sr, 0, marker="^", color=DOMAIN_COLORS[domain], s=120, label=f"{domain} (no real data)")
        elif real_sr is not None:
            ax.scatter(0, real_sr, marker="s", color=DOMAIN_COLORS[domain], s=120, label=f"{domain} (no sim data)")

    # diagonal
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 1.05)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, linewidth=1, label="y=x (equal)")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Simulated Agent (keyword match) — Full Context Success", fontsize=11)
    ax.set_ylabel("Real LLM — Full Context Success", fontsize=11)
    ax.set_title(
        "Full-Context Ceiling: Real LLM vs Simulated Agent",
        fontsize=12, fontweight="bold",
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "06_real_vs_simulated.png")


# ── Figure 7: Context reduction efficiency ────────────────────────────────────

def plot_efficiency_scatter(all_data: dict[str, dict]) -> None:
    """X = context reduction %, Y = success rate.  One point per (domain, condition)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    markers = {
        "full_context_test":          "o",
        "non_learning_manager_test":  "s",
        "lesson_aware_agent_a_test":  "^",
        "lesson_aware_agent_b_test":  "D",
    }

    for domain in DOMAIN_ORDER:
        modes   = all_data.get(domain, {})
        summary = modes.get("real") or modes.get("simulated")
        if summary is None:
            continue

        full_words = get_metric(summary, "full_context_test", "avg_active_context_words") or 1

        for cond in CONDITIONS:
            sr    = get_metric(summary, cond, "success_rate")
            words = get_metric(summary, cond, "avg_active_context_words")
            if sr is None or words is None:
                continue
            reduction = (full_words - words) / max(full_words, 1) * 100
            ax.scatter(
                reduction, sr,
                color=DOMAIN_COLORS[domain],
                marker=markers[cond],
                s=120, alpha=0.8, zorder=5,
            )

    # legend: domains (colour)
    for domain in DOMAIN_ORDER:
        ax.scatter([], [], color=DOMAIN_COLORS[domain], s=80,
                   label=DOMAIN_LABELS[domain].replace("\n", " "))
    ax.scatter([], [], marker="o", color="grey", s=80, label="Full Context")
    ax.scatter([], [], marker="s", color="grey", s=80, label="Non-Learning")
    ax.scatter([], [], marker="^", color="grey", s=80, label="Lesson-Aware A")
    ax.scatter([], [], marker="D", color="grey", s=80, label="Agent B (transfer)")

    ax.set_xlabel("Context Reduction vs Full Context (%)", fontsize=11)
    ax.set_ylabel("Task Success Rate", fontsize=11)
    ax.set_title(
        "Context Efficiency vs Task Success\n(All Conditions × All Domains)",
        fontsize=12, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "07_efficiency_scatter.png")


# ── Summary table ─────────────────────────────────────────────────────────────

def write_summary_table(all_data: dict[str, dict]) -> None:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("EXPERIENTIAL CONTEXT-MANAGEMENT AGENT — CROSS-DOMAIN RESULTS")
    lines.append("=" * 100)
    lines.append("")

    for mode_label, mode_key in [("Real LLM", "real"), ("Simulated Agent", "simulated")]:
        lines.append(f"Mode: {mode_label}")
        lines.append("-" * 100)
        hdr = f"{'Domain':<18} {'Ceiling':>8}  {'Baseline':>9}  {'LessonA':>8}  {'AgentB':>8}  "
        hdr += f"{'Lessons':>8}  {'Words(full)':>12}  {'Words(A)':>9}  {'Reduction':>10}"
        lines.append(hdr)
        lines.append("-" * 100)

        for domain in DOMAIN_ORDER:
            modes   = all_data.get(domain, {})
            summary = modes.get(mode_key)
            if summary is None:
                # fall back to incident (always real)
                if domain == "incident" and mode_key == "real":
                    summary = modes.get("real")
                if summary is None:
                    lines.append(f"  {domain:<16}  {'N/A':>8}  {'N/A':>9}  {'N/A':>8}  {'N/A':>8}")
                    continue

            ceil_   = get_metric(summary, "full_context_test",          "success_rate")
            base_   = get_metric(summary, "non_learning_manager_test",   "success_rate")
            les_a   = get_metric(summary, "lesson_aware_agent_a_test",   "success_rate")
            les_b   = get_metric(summary, "lesson_aware_agent_b_test",   "success_rate")
            w_full  = get_metric(summary, "full_context_test",           "avg_active_context_words")
            w_a     = get_metric(summary, "lesson_aware_agent_a_test",   "avg_active_context_words")
            n_les   = len(summary.get("learned_lessons_a", summary.get("lessons_learned", [])))
            red     = (w_full - w_a) / w_full * 100 if w_full and w_a else float("nan")

            row = f"  {domain:<16}  {_fmt_pct(ceil_):>8}  {_fmt_pct(base_):>9}  "
            row += f"{_fmt_pct(les_a):>8}  {_fmt_pct(les_b):>8}  "
            row += f"{n_les:>8}  {(w_full or 0):>12.1f}  {(w_a or 0):>9.1f}  "
            row += f"{red if red==red else float('nan'):>9.0f}%"
            lines.append(row)
        lines.append("")

    # combined summary JSON
    combined: dict = {}
    for domain in DOMAIN_ORDER:
        modes = all_data.get(domain, {})
        for mode_key in ("real", "simulated"):
            summary = modes.get(mode_key)
            if summary is None:
                continue
            key = f"{domain}_{mode_key}"
            combined[key] = {
                cond: {
                    "success_rate":               get_metric(summary, cond, "success_rate"),
                    "avg_active_context_words":   get_metric(summary, cond, "avg_active_context_words"),
                    "n":                          (summary.get("summary", {}).get(cond) or {}).get("n"),
                }
                for cond in CONDITIONS
            }
            combined[key]["lessons"] = len(
                summary.get("learned_lessons_a", summary.get("lessons_learned", []))
            )

    table_text = "\n".join(lines)
    (OUT_DIR / "summary_table.txt").write_text(table_text, encoding="utf-8")
    (OUT_DIR / "combined_summary.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    print(table_text)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading results from {RESULTS_DIR} ...")
    all_data = load_all()

    available = {
        d: {m: v is not None for m, v in modes.items()}
        for d, modes in all_data.items()
    }
    print(f"Available data: {available}")
    print(f"Output dir: {OUT_DIR}")
    print()

    print("Generating plots ...")
    plot_success_by_condition(all_data)
    plot_context_words(all_data)
    plot_learning_gain(all_data)
    plot_transfer(all_data)
    plot_family_heatmap(all_data)
    plot_real_vs_simulated(all_data)
    plot_efficiency_scatter(all_data)

    print("Writing summary table ...")
    write_summary_table(all_data)

    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
