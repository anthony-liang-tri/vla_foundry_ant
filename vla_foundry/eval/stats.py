"""Statistical functions for evaluation comparison (beta posteriors, CLD, z-tests).

All functions are pure — no database or filesystem dependencies.
Requires ``scipy`` and ``numpy`` (available via ``uv sync --group eval-viewer``).

Origin: Extracted from ``vla_foundry/tri/frontend/server.py`` (the internal
leaderboard server). The statistical functions (``compute_beta_params``,
``draw_beta_samples``, ``compute_beta_quantiles``, ``compare_two_proportions``,
``compute_cld``, ``compute_comparison_stats``) are identical to the originals.
``clopper_pearson_ci`` is adapted from
``vla_foundry/tri/lbm_eval/eval_campaigns/success_stats.py``
(``calculate_confidence_interval``), simplified to two-sided only.
The plot builders (``bar_chart``, ``violin_chart``, ``spider_chart``) are new.
"""

from __future__ import annotations


def clopper_pearson_ci(
    successes: int,
    total: int,
    confidence: float = 0.9,
) -> tuple[float, float]:
    """Clopper-Pearson confidence interval for a binomial proportion.

    Args:
        successes: Number of successes.
        total: Total number of trials.
        confidence: Confidence level (0 to 1). Default 0.9 matches the
            internal leaderboard.

    Returns:
        ``(lower_bound, upper_bound)`` of the confidence interval.
    """
    if total == 0:
        return 0.0, 1.0
    from scipy.stats import binomtest

    r = binomtest(successes, total).proportion_ci(confidence)
    return r.low, r.high


def compute_beta_params(successes: int, total: int, alpha_prior: float = 1, beta_prior: float = 1):
    """Beta distribution parameters from success counts."""
    alpha = alpha_prior + successes
    beta = beta_prior + (total - successes)
    return alpha, beta


def compute_beta_quantiles(alpha: float, beta: float, quantiles: list[float]) -> list[float]:
    """Quantiles of a Beta(alpha, beta) distribution."""
    from scipy import stats

    dist = stats.beta(alpha, beta)
    return [float(dist.ppf(q)) for q in quantiles]


def draw_beta_samples(alpha: float, beta: float, n_samples: int = 1000, seed: int = 42) -> list[float]:
    """Draw random samples from Beta(alpha, beta) for violin plots."""
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(seed)
    return stats.beta(alpha, beta).rvs(n_samples, random_state=rng).tolist()


def compare_two_proportions(n_a: int, k_a: int, n_b: int, k_b: int, alpha: float = 0.05) -> int:
    """Two-proportion z-test.  Returns 1 if a>b, -1 if a<b, 0 if not significant."""
    import numpy as np
    from scipy import stats

    if n_a == 0 or n_b == 0:
        return 0
    p_a, p_b = k_a / n_a, k_b / n_b
    p_pooled = (k_a + k_b) / (n_a + n_b)
    if p_pooled == 0 or p_pooled == 1:
        return 0
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0
    z = (p_a - p_b) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    if p_value < alpha:
        return 1 if p_a > p_b else -1
    return 0


def compute_cld(results_list: list[dict], alpha: float = 0.05) -> dict[str, str]:
    """Compact Letter Display for statistical groupings.

    *results_list* items must have ``label``, ``successes``, and ``total`` keys.
    Returns ``{label: letter(s)}``.
    """
    labels = [r["label"] for r in results_list]
    if len(labels) < 2:
        return {labels[0]: "a"} if labels else {}

    sig_pairs: list[tuple[str, str]] = []
    for i in range(len(results_list)):
        for j in range(i + 1, len(results_list)):
            r_i, r_j = results_list[i], results_list[j]
            if r_i["total"] == 0 or r_j["total"] == 0:
                continue
            if compare_two_proportions(r_i["total"], r_i["successes"], r_j["total"], r_j["successes"], alpha) != 0:
                sig_pairs.append((labels[i], labels[j]))

    sorted_labels = [
        p[0]
        for p in sorted(
            ((r["label"], r["successes"] / r["total"] if r["total"] > 0 else 0) for r in results_list),
            key=lambda x: x[1],
            reverse=True,
        )
    ]

    cld: dict[str, str] = {}
    current_letter = ord("a")
    assigned: set[str] = set()
    for label in sorted_labels:
        if label in assigned:
            continue
        group = [label] + [
            other
            for other in sorted_labels
            if other != label
            and other not in assigned
            and (label, other) not in sig_pairs
            and (other, label) not in sig_pairs
        ]
        letter = chr(current_letter)
        for g in group:
            cld[g] = cld.get(g, "") + letter
        assigned.update(group)
        current_letter += 1
    return cld


_STANDARD_QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]


def _beta_entry(successes: int, total: int, n_samples: int, **extra) -> dict:
    """Build a single result entry with beta params, samples, and quantiles."""
    alpha, beta = compute_beta_params(successes, total)
    samples = draw_beta_samples(alpha, beta, n_samples)
    quantiles = compute_beta_quantiles(alpha, beta, _STANDARD_QUANTILES)
    return {
        **extra,
        "successes": successes,
        "total": total,
        "success_rate": successes / total if total > 0 else 0,
        "alpha": alpha,
        "beta": beta,
        "violin_samples": samples,
        "quantiles": {
            "q025": quantiles[0],
            "q25": quantiles[1],
            "median": quantiles[2],
            "q75": quantiles[3],
            "q975": quantiles[4],
        },
    }


def compute_comparison_stats(
    entries: list[dict],
    n_samples: int = 500,
) -> dict:
    """Compute violin/CLD comparison data from a list of per-(task, model) entries.

    Each entry must have ``task_name``, ``label`` (model/ablation name),
    ``successes``, and ``total``.

    Returns ``{by_task, task_order, cld_by_task, aggregate, aggregate_cld}``.
    """
    import numpy as np

    by_task: dict[str, list[dict]] = {}
    all_results: list[dict] = []

    for e in entries:
        result = _beta_entry(
            e["successes"],
            e["total"],
            n_samples,
            task_name=e["task_name"],
            label=e["label"],
            ablation=e["label"],
        )
        by_task.setdefault(e["task_name"], []).append(result)
        all_results.append(result)

    # CLD per task
    cld_by_task: dict[str, dict[str, str]] = {}
    for task, task_results in by_task.items():
        cld = compute_cld(task_results)
        cld_by_task[task] = cld
        for r in task_results:
            r["cld_letter"] = cld.get(r["label"], "")

    # Aggregate per label (model/ablation)
    agg_map: dict[str, dict[str, int]] = {}
    for r in all_results:
        lbl = r["label"]
        if lbl not in agg_map:
            agg_map[lbl] = {"successes": 0, "total": 0}
        agg_map[lbl]["successes"] += r["successes"]
        agg_map[lbl]["total"] += r["total"]

    aggregate = [
        _beta_entry(a["successes"], a["total"], n_samples, label=lbl, ablation=lbl) for lbl, a in agg_map.items()
    ]
    agg_cld = compute_cld(aggregate)
    for r in aggregate:
        r["cld_letter"] = agg_cld.get(r["label"], "")

    task_order = sorted(
        by_task.keys(),
        key=lambda t: float(np.mean([r["success_rate"] for r in by_task[t]])),
        reverse=True,
    )

    return {
        "by_task": by_task,
        "task_order": task_order,
        "cld_by_task": cld_by_task,
        "aggregate": aggregate,
        "aggregate_cld": agg_cld,
    }


# ---------------------------------------------------------------------------
# Plot builders (Plotly figures from aggregated stats)
# ---------------------------------------------------------------------------

COLORS = [
    "#4DBBD5",
    "#E64B35",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#7E6148",
    "#B09C85",
]


def _with_alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def bar_chart(data: list[dict]):
    """Grouped bar chart of success rates by task with 90% CIs."""
    import numpy as np
    import plotly.graph_objects as go

    if not data:
        return go.Figure().update_layout(title="No data")
    models = sorted({s["model"] for s in data})
    tasks = sorted(
        {s["task"] for s in data},
        key=lambda t: -max((s["pct"] for s in data if s["task"] == t), default=0),
    )
    lookup = {(s["task"], s["model"]): s for s in data}
    fig = go.Figure()
    for i, model in enumerate(models):
        rates, err_lo, err_hi = [], [], []
        for task in tasks:
            s = lookup.get((task, model))
            if s:
                rates.append(s["pct"])
                err_lo.append(s["pct"] - s["ci_low"] * 100)
                err_hi.append(s["ci_high"] * 100 - s["pct"])
            else:
                rates.append(None)
                err_lo.append(0)
                err_hi.append(0)
        fig.add_trace(
            go.Bar(
                name=model,
                x=tasks,
                y=rates,
                error_y=dict(type="data", symmetric=False, array=np.array(err_hi), arrayminus=np.array(err_lo)),
                marker_color=COLORS[i % len(COLORS)],
            )
        )
    fig.update_layout(
        title="Success Rate by Task (90% CI)",
        yaxis_title="Success Rate (%)",
        yaxis_range=[0, 105],
        barmode="group",
        xaxis_tickangle=-45,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(b=150, t=60),
        plot_bgcolor="white",
    )
    return fig


def violin_chart(data: list[dict]):
    """Beta posterior violin plots with CLD significance letters."""
    import numpy as np
    import plotly.graph_objects as go

    if not data:
        return go.Figure().update_layout(title="No data")
    entries = [
        {"task_name": s["task"], "label": s["model"], "successes": s["successes"], "total": s["total"]} for s in data
    ]
    comp = compute_comparison_stats(entries, n_samples=500)
    by_task, task_order, aggregate = comp["by_task"], comp["task_order"], comp["aggregate"]
    all_labels = sorted({r["label"] for v in by_task.values() for r in v})
    cmap = {a: COLORS[i % len(COLORS)] for i, a in enumerate(all_labels)}
    n = len(all_labels)
    gw, vw = 0.8, min(0.35, 0.8 / max(n, 1))

    fig = go.Figure()
    shown: set[str] = set()
    annotations: list[dict] = []

    def _add(samples, label, xp, legend, width):
        fig.add_trace(
            go.Violin(
                y=np.array(samples),
                x=np.full(len(samples), xp),
                name=label,
                legendgroup=label,
                showlegend=legend,
                scalegroup="all",
                points=False,
                box_visible=False,
                meanline_visible=True,
                line_color=cmap[label],
                fillcolor=cmap[label],
                opacity=0.7,
                width=width,
                hoverinfo="skip",
            )
        )

    for ti, task in enumerate(task_order):
        for r in by_task.get(task, []):
            samples = r.get("violin_samples", [])
            if not samples:
                continue
            ai = all_labels.index(r["label"])
            xp = ti + (ai - (n - 1) / 2) * (gw / max(n, 1))
            _add(samples, r["label"], xp, r["label"] not in shown, vw)
            shown.add(r["label"])
            cld = r.get("cld_letter", "")
            if cld and n > 1:
                annotations.append(
                    dict(
                        x=xp,
                        y=min(r["success_rate"] + 0.08, 1.02),
                        text=f"<b>{cld}</b>",
                        showarrow=False,
                        font=dict(size=10),
                    )
                )

    agg_x = len(task_order) + 1
    for agg in aggregate:
        samples = agg.get("violin_samples", [])
        if not samples:
            continue
        ai = all_labels.index(agg["label"])
        xp = agg_x + (ai - (n - 1) / 2) * (gw / max(n, 1))
        _add(samples, agg["label"], xp, False, vw * 1.5)
        cld = agg.get("cld_letter", "")
        if cld and n > 1:
            annotations.append(
                dict(
                    x=xp,
                    y=min(agg["success_rate"] + 0.08, 1.02),
                    text=f"<b>{cld}</b>",
                    showarrow=False,
                    font=dict(size=10),
                )
            )

    ticks = list(range(len(task_order))) + [agg_x]
    tick_text = [t[:25] + "\u2026" if len(t) > 25 else t for t in task_order] + ["Aggregate"]
    fig.update_layout(
        title="Success Rate Distribution (Beta Posterior)",
        yaxis=dict(title="Success Rate", range=[-0.02, 1.15]),
        xaxis=dict(tickangle=-45, tickvals=ticks, ticktext=tick_text, range=[-0.7, agg_x + 0.7]),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        annotations=annotations,
        shapes=[
            dict(
                type="line",
                x0=len(task_order) - 0.15,
                x1=len(task_order) - 0.15,
                y0=-0.02,
                y1=1.1,
                line=dict(color="gray", width=2),
            )
        ]
        if task_order
        else [],
        margin=dict(b=150, t=60, l=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
        violingap=0,
        violinmode="overlay",
    )
    return fig


def spider_chart(data: list[dict]):
    """Radar chart of success rates by task and model."""
    import plotly.graph_objects as go

    if not data:
        return go.Figure().update_layout(title="No data")
    models = sorted({s["model"] for s in data})
    tasks = sorted({s["task"] for s in data})
    lookup = {(s["task"], s["model"]): s for s in data}
    short = [t[:22] + "\u2026" if len(t) > 22 else t for t in tasks]
    fig = go.Figure()
    for i, model in enumerate(models):
        rates = [lookup.get((t, model), {}).get("pct", 0) for t in tasks]
        fig.add_trace(
            go.Scatterpolar(
                r=rates + [rates[0]],
                theta=short + [short[0]],
                fill="toself",
                name=model,
                line_color=COLORS[i % len(COLORS)],
                fillcolor=_with_alpha(COLORS[i % len(COLORS)], 0.2),
            )
        )
    fig.update_layout(
        title="Success Rate by Task",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%"), angularaxis=dict(direction="clockwise")
        ),
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
        margin=dict(t=60, b=60, l=80, r=80),
        paper_bgcolor="white",
    )
    return fig
