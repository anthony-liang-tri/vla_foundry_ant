"""Statistical functions for evaluation comparison (beta posteriors, CLD, STEP test).

All functions are pure — no database or filesystem dependencies.
Requires ``scipy``, ``numpy``, and ``sequentialized-barnard-tests``
(available via ``uv sync --group eval-viewer``).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from scipy.stats import binomtest
from sequentialized_barnard_tests import Decision, Hypothesis, MirroredLaiTest, MirroredStepTest
from sequentialized_barnard_tests.auto import get_mirrored_test
from sequentialized_barnard_tests.tools.plotting import compact_letter_display, draw_samples_from_beta_posterior


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

    r = binomtest(successes, total).proportion_ci(confidence)
    return r.low, r.high


def build_success_arrays(
    episodes: list[dict],
) -> dict:
    """Convert raw episodes to per-(task, model) boolean arrays.

    Returns ``{task: {model: array}, "__aggregate__": {model: concatenated}}``.
    The ``"__aggregate__"`` entry concatenates each model's arrays across all tasks.
    """
    data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for ep in episodes:
        data[ep["task"]][ep["model"]].append(bool(ep["success"]))

    result: dict[str, dict[str, np.ndarray]] = {}
    agg: dict[str, list] = defaultdict(list)

    for task, model_dict in data.items():
        result[task] = {}
        for model, bools in model_dict.items():
            result[task][model] = np.array(bools, dtype=bool)
            agg[model].extend(bools)

    result["__aggregate__"] = {m: np.array(v, dtype=bool) for m, v in agg.items()}
    return result


def compute_cld_step(
    success_arrays_by_task: dict,
    max_sample_size_per_model: int,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[dict[str, dict[str, str]], str]:
    """Compute per-task and aggregate CLD using the STEP sequential test.

    Calls ``compare_success_and_get_cld_auto`` with Bonferroni correction
    across all pairwise comparisons.

    Per-task comparisons use ``shuffle=False``; the aggregate uses ``shuffle=True``
    (episodes from different tasks are concatenated, so order should be shuffled).

    Args:
        success_arrays_by_task: Output of ``build_success_arrays``.
        max_sample_size_per_model: Maximum number of rollouts per model. Must be
            set based on the experimental budget *before* collecting data.
            ``load_episodes`` enforces that all results files share the same
            value, so this is always the recorded pre-commitment budget.
        confidence_level: Global confidence level (default 0.95).
        seed: RNG seed for the aggregate shuffle.

    Returns:
        A tuple ``(cld_by_task, warning_msg)``.  ``cld_by_task`` maps each task
        key (including ``"__aggregate__"``) to ``{model: cld_letter}``.
        ``warning_msg`` is non-empty if any array length exceeds
        ``max_sample_size_per_model``.
    """
    # Check for sample size violations (exclude aggregate since it's derived).
    violations: list[tuple[str, str, int]] = []
    all_task_lengths: list[int] = []
    for task, model_dict in success_arrays_by_task.items():
        if task == "__aggregate__":
            continue
        for model, arr in model_dict.items():
            all_task_lengths.append(len(arr))
            if len(arr) > max_sample_size_per_model:
                violations.append((task, model, len(arr)))

    warning_msg = ""
    if violations:
        ex_task, ex_model, ex_len = violations[0]
        warning_msg = (
            f"Some rollout counts exceed the pre-committed `max_sample_size` "
            f'(e.g. task "{ex_task}", model "{ex_model}": {ex_len} > '
            f"{max_sample_size_per_model}). "
            f"Only the first **{max_sample_size_per_model}** observations "
            f"per model are used for statistical testing. "
            f"Extra observations are ignored to preserve the validity of "
            f"the sequential test's error guarantees."
        )

    cld_by_task: dict[str, dict[str, str]] = {}
    rng = np.random.default_rng(seed)
    num_tasks = sum(1 for k in success_arrays_by_task if k != "__aggregate__")

    for task, model_dict in success_arrays_by_task.items():
        models = sorted(model_dict.keys())
        if not models:
            cld_by_task[task] = {}
            continue
        if len(models) == 1:
            cld_by_task[task] = {models[0]: "a"}
            continue

        is_agg = task == "__aggregate__"
        # For the aggregate test the maximum sequence length is
        # max_sample_size_per_model * num_tasks (one budget per task).
        n_max = max_sample_size_per_model * num_tasks if is_agg else max_sample_size_per_model
        # Copy arrays — compare_success_and_get_cld shuffles them in-place.
        arrays = [model_dict[m].copy() for m in models]

        try:
            cld = compare_success_and_get_cld_auto(
                models,
                arrays,
                confidence_level,
                n_max,
                shuffle=is_agg,
                rng=rng if is_agg else None,
                verbose=False,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "CLD computation failed for task %r; marking as '?'",
                task,
                exc_info=True,
            )
            cld = {m: "?" for m in models}

        cld_by_task[task] = cld

    return cld_by_task, warning_msg


def compare_success_and_get_cld_auto(
    model_name_list: list[str],  # [model_0, ...]
    success_array_list: list[np.ndarray],  # [success_array_for_model_0, ...]
    global_confidence_level: float,
    n_max: int,
    shuffle: bool,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> dict[str, str]:
    """Compares multiple success arrays and returns their Compact Letter Display (CLD)
    representation based on pairwise sequential tests (STEP or Lai, auto-selected).

    Args:
        model_name_list: A list of model names.
        success_array_list: A list of binary arrays indicating success/failure
            for each model.
        global_confidence_level: The desired global confidence level for the
            multiple comparisons.
        n_max: The maximum sequence length for the sequential test. For
            per-task comparisons this equals ``max_sample_size_per_model``; for
            aggregate comparisons it equals
            ``max_sample_size_per_model * num_tasks``.
        shuffle: Whether to shuffle the True/False ordering of each success array
            before comparison. Set it to False if each True/False outcome is
            independent within each array. Set to True if, for example, each array is a
            concatenation of results from multiple tasks and you want to measure the
            aggregate multi-task performance.
        rng: Optional random number generator instance for shuffling. Only used if
            shuffle is True.
        verbose: Whether to print detailed output. Defaults to True.
    Returns:
        A dictionary mapping model names to their CLD letters.
    """
    if shuffle and rng is None:
        raise ValueError("rng must be provided when shuffle is True.")
    num_models = len(model_name_list)
    # Set up the sequential statistical test.
    global_alpha = 1 - global_confidence_level
    num_comparisons = num_models * (num_models - 1) // 2
    individual_alpha = global_alpha / num_comparisons
    individual_confidence_level = 1 - individual_alpha
    test = get_mirrored_test(
        alternative=Hypothesis.P0LessThanP1,
        alpha=individual_alpha,
        n_max=n_max,
    )
    if verbose:
        if isinstance(test, MirroredStepTest):
            method_name = "STEP"
        elif isinstance(test, MirroredLaiTest):
            method_name = "Lai"
        else:
            method_name = type(test).__name__
        print("Statistical Test Specs:")
        print(f"  Method: {method_name}")
        print(f"  Global Confidence: {round(global_confidence_level, 5)}")
        print(f"    ({round(individual_confidence_level, 5)} per comparison)")
        print(f"  Maximum Sequence Length (n_max): {n_max}\n")

    # No explicit reset needed — run_on_sequence() calls reset() internally.

    # Prepare success array per model.
    success_array_dict = dict()  # model_name -> success_array
    for idx in np.arange(num_models):
        model = model_name_list[idx]
        success_array = success_array_list[idx]
        if shuffle:
            rng.shuffle(success_array)
        success_array_dict[model] = success_array

    # Run pairwise comparisons.
    comparisons_dict = dict()  # (model_name_a, model_name_b) -> Decision
    for idx_a in np.arange(num_models):
        for idx_b in np.arange(idx_a + 1, num_models):
            model_a = model_name_list[idx_a]
            model_b = model_name_list[idx_b]
            array_a = success_array_dict[model_a]
            array_b = success_array_dict[model_b]
            len_common = min(len(array_a), len(array_b))
            array_a = array_a[:len_common]
            array_b = array_b[:len_common]
            # Run the test.
            test_result = test.run_on_sequence(array_a, array_b)
            comparisons_dict[(model_a, model_b)] = test_result.decision

    # Compact Letter Display algorithm to summarize results
    input_list_to_cld = list()
    for key, val in comparisons_dict.items():
        if val != Decision.FailToDecide:
            input_list_to_cld.append(key)
    models_sorted_by_success_rates = [
        model
        for model, _ in sorted(
            success_array_dict.items(),
            key=lambda kv_pair: np.mean(kv_pair[1]) if len(kv_pair[1]) else 0.0,
            reverse=True,
        )
    ]
    letters_list = compact_letter_display(input_list_to_cld, models_sorted_by_success_rates)
    if verbose:
        print("Statistical Test Results (Compact Letter Display):")
    str_padding = max([len(model) for model in models_sorted_by_success_rates])
    return_dict = dict()
    for letters, model in zip(letters_list, models_sorted_by_success_rates, strict=True):
        return_dict[model] = letters
        num_successes = np.sum(success_array_dict[model])
        num_trials = len(success_array_dict[model])
        empirical_success_rate = 0.0 if num_trials == 0 else np.mean(success_array_dict[model])
        if verbose:
            print(
                f"  CLD for {model:<{str_padding}}: {letters}\n"
                f"    Success Rate {num_successes} / {num_trials} = "
                f"{round(empirical_success_rate, 3)}",
            )

    # Ranks are determined if each policy has a unique single letter.
    all_order_determined = all([len(letters) == 1 for letters in letters_list]) and len(set(letters_list)) == len(
        model_name_list
    )
    if verbose:
        if all_order_determined:
            print(f"All models separated with global confidence of {round(global_confidence_level, 5)}.")
        else:
            print(
                "Not all models were separated with global confidence of "
                f"{round(global_confidence_level, 5)}. Models that share "
                "a same letter are not separated from each other with "
                "statistical significance. For more information on how to "
                "interpret the letters, see: "
                "https://en.wikipedia.org/wiki/Compact_letter_display.\n"
            )
    return return_dict


# ---------------------------------------------------------------------------
# Plot builders (Plotly figures from raw episodes)
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


def model_comparison_chart(
    episodes: list[dict],
    max_sample_size_per_model: int | None = None,
    overlay_on_bars: bool = False,
    confidence_level: float = 0.95,
    n_posterior_samples: int = 2000,
    seed: int = 42,
) -> tuple:
    """Plotly adaptation of ``plot_model_comparison`` from sequentialized_barnard_tests.

    Draws beta posterior violin plots per (task, model) with:
    - Horizontal mean lines through each violin (``meanline_visible=True``).
    - Empirical mean dots (black circle markers).
    - CLD letter annotations above each violin's posterior mean (STEP test,
      Bonferroni-corrected). Only shown when ``max_sample_size_per_model`` is set.
    - Optional semi-transparent bar overlay showing the empirical mean height.
    - A vertical separator before the aggregate section.

    Args:
        episodes: Raw episode dicts with ``task``, ``model``, and ``success`` keys.
        max_sample_size_per_model: The ``max_sample_size_per_model`` value from
            the results files (returned by ``load_episodes``). Used as the STEP
            test budget. When ``None``, violins are shown without CLD annotations.
        overlay_on_bars: If ``True``, draw semi-transparent bars behind each
            violin showing the empirical mean.
        confidence_level: Global confidence level for CLD (default 0.95).
        n_posterior_samples: Number of samples drawn from the beta posterior for
            each violin (default 2000).
        seed: RNG seed for posterior sampling and aggregate shuffling.

    Returns:
        ``(figure, warning_msg)``.  ``warning_msg`` is non-empty when any
        rollout count exceeds ``max_sample_size_per_model``.
    """

    if not episodes:
        return go.Figure().update_layout(title="No data"), ""

    rng = np.random.default_rng(seed)
    arrays_by_task = build_success_arrays(episodes)

    pure_tasks = sorted(k for k in arrays_by_task if k != "__aggregate__")
    all_models = sorted({m for task_dict in arrays_by_task.values() for m in task_dict})

    cld_by_task: dict[str, dict[str, str]] = {}
    warning_msg = ""
    if max_sample_size_per_model is not None:
        cld_by_task, warning_msg = compute_cld_step(
            arrays_by_task, max_sample_size_per_model, confidence_level, seed + 1
        )

    task_order = sorted(
        pure_tasks,
        key=lambda t: float(np.mean([np.mean(arr) for arr in arrays_by_task[t].values()])),
        reverse=True,
    )

    n = len(all_models)
    cmap = {m: COLORS[i % len(COLORS)] for i, m in enumerate(all_models)}
    gw = 0.8
    vw = min(0.35, 0.8 / max(n, 1))

    fig = go.Figure()
    shown: set[str] = set()
    annotations: list[dict] = []

    def _add_violin(xp: float, model: str, arr: np.ndarray, cld_key: str, width: float) -> None:
        samples = draw_samples_from_beta_posterior(arr, rng, n_posterior_samples)
        posterior_mean = float(np.mean(samples))
        empirical_mean = float(np.mean(arr))
        color = cmap[model]

        if overlay_on_bars:
            fig.add_trace(
                go.Bar(
                    x=[xp],
                    y=[empirical_mean],
                    width=width * 1.2,
                    marker_color=_with_alpha(color, 0.3),
                    marker_line_width=0,
                    showlegend=False,
                    legendgroup=model,
                    hoverinfo="skip",
                )
            )

        fig.add_trace(
            go.Violin(
                y=samples,
                x=np.full(len(samples), xp),
                name=model,
                legendgroup=model,
                showlegend=model not in shown,
                scalegroup="all",
                points=False,
                box_visible=False,
                meanline=dict(visible=True, color="black", width=1.5),
                line_color=color,
                fillcolor=color,
                opacity=0.7,
                width=width,
                hoverinfo="skip",
            )
        )
        shown.add(model)

        # Empirical mean dot
        fig.add_trace(
            go.Scatter(
                x=[xp],
                y=[empirical_mean],
                mode="markers",
                marker=dict(color="black", size=6, symbol="circle"),
                legendgroup=model,
                showlegend=False,
                hoverinfo="skip",
            )
        )

        cld_letter = cld_by_task.get(cld_key, {}).get(model, "")
        if cld_letter and n > 1:
            annotations.append(
                dict(
                    x=xp,
                    y=min(posterior_mean + 0.08, 1.02),
                    text=f"<b>{cld_letter}</b>",
                    showarrow=False,
                    font=dict(size=10),
                )
            )

    for ti, task in enumerate(task_order):
        for model in sorted(arrays_by_task.get(task, {}).keys()):
            ai = all_models.index(model)
            xp = ti + (ai - (n - 1) / 2) * (gw / max(n, 1))
            _add_violin(xp, model, arrays_by_task[task][model], task, vw)

    agg_x = len(task_order) + 1
    agg = arrays_by_task.get("__aggregate__", {})
    for model in sorted(agg.keys()):
        ai = all_models.index(model)
        xp = agg_x + (ai - (n - 1) / 2) * (gw / max(n, 1))
        _add_violin(xp, model, agg[model], "__aggregate__", vw * 1.5)

    ticks = list(range(len(task_order))) + [agg_x]
    tick_text = [t[:25] + "\u2026" if len(t) > 25 else t for t in task_order] + ["Aggregate"]

    fig.update_layout(
        title="Success Rate Distribution (Beta Posterior)",
        yaxis=dict(title="Success Rate", range=[-0.02, 1.15]),
        xaxis=dict(
            tickangle=-45,
            tickvals=ticks,
            ticktext=tick_text,
            range=[-0.7, agg_x + 0.7],
        ),
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
        barmode="overlay",
    )

    return fig, warning_msg


def spider_chart(data: list[dict]):
    """Radar chart of success rates by task and model."""
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
