"""Gradio dashboard for simulation evaluation results.

Reuses ``data_loading`` for filesystem scanning and ``stats`` for
statistical comparisons (beta posteriors, CLD significance letters via
the STEP sequential test from ``sequentialized_barnard_tests``).

Usage::

    uv run --group dashboard python vla_foundry/eval/results_explorer.py rollouts/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from vla_foundry.eval.data_loading import aggregate_episodes, load_episodes
from vla_foundry.eval.stats import (
    clopper_pearson_ci,
    model_comparison_chart,
    spider_chart,
)


def _scan(root: Path):
    eps, pending_by, crashed_by, max_sample_size = load_episodes(root)
    stats = aggregate_episodes(eps, ci_fn=clopper_pearson_ci, pending_by=pending_by, crashed_by=crashed_by)
    return eps, stats, max_sample_size


# ---------------------------------------------------------------------------
# Data builders (return DataFrames / markdown for native Gradio components)
# ---------------------------------------------------------------------------


def _stats_markdown(stats: list[dict]) -> str:
    if not stats:
        return "*No data*"
    total = sum(s["total"] for s in stats)
    return (
        f"**{len({s['task'] for s in stats})}** Tasks \u00a0\u00b7\u00a0 "
        f"**{len({s['model'] for s in stats})}** Models \u00a0\u00b7\u00a0 "
        f"**{total:,}** Episodes"
    )


def _summary_df(data: list[dict]):
    if not data:
        return pd.DataFrame().style
    rows = []
    for s in sorted(data, key=lambda x: x["pct"], reverse=True):
        rows.append(
            {
                "Task": s["task"],
                "Model": s["model"],
                "Success Rate": s["pct"],
                "Successes": s["successes"],
                "Completed Rollouts": s["total"],
                "Pending": s.get("pending", 0),
                "Crashed": s.get("crashed", 0),
                "90% CI": f"[{s['ci_low']:.1%}, {s['ci_high']:.1%}]",
                "Avg Duration": f"{s['avg_dur']:.1f}s",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def _rate_bg(pct):
        """Return a dark-mode-friendly background color for a 0-100 percentage."""
        if pct >= 75:
            return "background-color: rgba(40, 167, 69, 0.25)"
        if pct >= 50:
            return "background-color: rgba(255, 193, 7, 0.25)"
        return "background-color: rgba(220, 53, 69, 0.25)"

    def _color_row(row):
        success_bg = _rate_bg(row["Success Rate"])
        total = row["Completed Rollouts"] + row["Pending"] + row["Crashed"]
        completion_pct = 100.0 * row["Completed Rollouts"] / total if total > 0 else 0
        completion_bg = _rate_bg(completion_pct)
        return [
            (
                success_bg
                if col in ("Success Rate", "Successes")
                else completion_bg
                if col == "Completed Rollouts"
                else ""
            )
            for col in row.index
        ]

    styler = df.style.apply(_color_row, axis=1).format({"Success Rate": "{:.1f}%"})
    return styler


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------


def create_app(root: Path):
    """Build the Gradio application."""
    import gradio as gr

    gr.set_static_paths(paths=[str(root)])

    state: dict = {}
    state["episodes"], state["stats"], state["max_sample_size"] = _scan(root)

    def all_tasks():
        return sorted({s["task"] for s in state["stats"]})

    def all_models():
        return sorted({s["model"] for s in state["stats"]})

    def _filtered(tasks_sel, models_sel):
        ts, ms = set(tasks_sel or []), set(models_sel or [])
        return [s for s in state["stats"] if s["task"] in ts and s["model"] in ms]

    def _parse_episode_filter(episode_filter: str) -> set[str] | None:
        """Parse comma-separated episode/demo IDs. Returns None if no filter."""
        if not episode_filter or not episode_filter.strip():
            return None
        return {tok.strip() for tok in episode_filter.split(",") if tok.strip()}

    def _filtered_eps(tasks_sel, models_sel, outcome="All", episode_filter=""):
        ts, ms = set(tasks_sel or []), set(models_sel or [])
        eps = [ep for ep in state["episodes"] if ep["task"] in ts and ep["model"] in ms]
        if outcome == "Success":
            eps = [ep for ep in eps if ep["success"]]
        elif outcome == "Failure":
            eps = [ep for ep in eps if not ep["success"]]
        demo_ids = _parse_episode_filter(episode_filter)
        if demo_ids is not None:
            eps = [ep for ep in eps if str(ep["demo_id"]) in demo_ids]
        return eps

    GRID_SIZE = 9  # 3x3

    def _rec_eps(tasks_sel, models_sel, outcome="All", episode_filter=""):
        return [ep for ep in _filtered_eps(tasks_sel, models_sel, outcome, episode_filter) if ep.get("recording_video")]

    def _grid_updates(tasks_sel, models_sel, outcome, page, episode_filter=""):
        """Return updates for the 9 video slots + link slots given filters and page."""
        eps = _rec_eps(tasks_sel, models_sel, outcome, episode_filter)
        total_pages = max(1, (len(eps) + GRID_SIZE - 1) // GRID_SIZE)
        page = max(0, min(page, total_pages - 1))
        page_eps = eps[page * GRID_SIZE : (page + 1) * GRID_SIZE]
        vid_updates = []
        link_updates = []
        for i in range(GRID_SIZE):
            if i < len(page_eps):
                ep = page_eps[i]
                badge = "\u2705" if ep["success"] else "\u274c"
                label = f"{ep['model']} | {ep['task']} | demo {ep['demo_id']} | {badge} | {ep['duration']:.1f}s"
                vid_updates.append(gr.update(value=ep["recording_video"], visible=True, label=label))
                html_path = ep.get("recording_html", "")
                if html_path:
                    link_updates.append(
                        gr.update(
                            value=f"[Open 3D Replay](/gradio_api/file={html_path})",
                            visible=True,
                        )
                    )
                else:
                    link_updates.append(gr.update(value="", visible=True))
            else:
                vid_updates.append(gr.update(value=None, visible=False))
                link_updates.append(gr.update(value="", visible=False))
        page_text = f"Page {page + 1} / {total_pages}  ({len(eps)} recordings)"
        return vid_updates, link_updates, page, page_text

    # ---- Shared filter output builder ----

    def _comparison_meta_text(comp_warning: str = ""):
        mss = state.get("max_sample_size")
        parts: list[str] = []
        if mss:
            parts.append(f"**Max sample size per model per task:** {mss}")
        else:
            parts.append("*No max sample size found in results -- CLD annotations will not be shown.*")
        if comp_warning:
            parts.append(comp_warning)
        return "\n\n".join(parts)

    def _build_filter_outputs(tasks_sel, models_sel, bar_overlay, outcome="All", episode_filter=""):
        """Shared helper that builds all filter-dependent outputs.

        CLD is computed automatically when ``max_sample_size_per_model`` is
        available in the loaded results.
        """
        d = _filtered(tasks_sel, models_sel)
        eps = _filtered_eps(tasks_sel, models_sel)
        mss = state.get("max_sample_size")
        comp_fig, comp_warning = model_comparison_chart(eps, mss, bool(bar_overlay))

        vids, links, page, page_text = _grid_updates(tasks_sel, models_sel, outcome, 0, episode_filter)
        return (
            _stats_markdown(d),
            _summary_df(d),
            _comparison_meta_text(comp_warning),
            comp_fig,
            spider_chart(d),
            *vids,
            *links,
            page,
            page_text,
        )

    # ---- Event handlers ----

    def on_filter(tasks_sel, models_sel, bar_overlay, outcome, episode_filter):
        return _build_filter_outputs(tasks_sel, models_sel, bar_overlay, outcome, episode_filter)

    def on_rec_filter(tasks_sel, models_sel, outcome, _page_state, episode_filter):
        vids, links, page, page_text = _grid_updates(tasks_sel, models_sel, outcome, 0, episode_filter)
        return (*vids, *links, page, page_text)

    def on_page(tasks_sel, models_sel, outcome, page_state, episode_filter, direction):
        new_page = page_state + direction
        vids, links, page, page_text = _grid_updates(tasks_sel, models_sel, outcome, new_page, episode_filter)
        return (*vids, *links, page, page_text)

    def on_select_all(bar_overlay, outcome, episode_filter):
        import gradio as gr

        tasks, models = all_tasks(), all_models()
        outputs = _build_filter_outputs(tasks, models, bar_overlay, outcome, episode_filter)
        return (
            gr.update(value=tasks),
            gr.update(value=models),
            *outputs,
        )

    def on_csv_download():
        d = state["stats"]
        df = pd.DataFrame(
            [
                {
                    "Task": s["task"],
                    "Model": s["model"],
                    "Success Rate": f"{s['pct']:.1f}%",
                    "Successes": s["successes"],
                    "Completed Rollouts": s["total"],
                    "Pending": s.get("pending", 0),
                    "Crashed": s.get("crashed", 0),
                    "90% CI Low": f"{s['ci_low']:.1%}",
                    "90% CI High": f"{s['ci_high']:.1%}",
                    "Avg Duration": f"{s['avg_dur']:.1f}s",
                }
                for s in sorted(d, key=lambda x: x["pct"], reverse=True)
            ]
        )
        csv_path = root / ".summary.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def on_refresh(bar_overlay, outcome, episode_filter):
        import gradio as gr

        state["episodes"], state["stats"], state["max_sample_size"] = _scan(root)
        tasks, models = all_tasks(), all_models()
        outputs = _build_filter_outputs(tasks, models, bar_overlay, outcome, episode_filter)
        return (
            gr.update(choices=tasks, value=tasks),
            gr.update(choices=models, value=models),
            *outputs,
        )

    # ---- Build UI ----

    init_tasks, init_models = all_tasks(), all_models()
    init_d = _filtered(init_tasks, init_models)
    init_eps = _filtered_eps(init_tasks, init_models)
    init_comp_fig, init_comp_warning = model_comparison_chart(init_eps, state.get("max_sample_size"))
    init_vids, init_links, init_page, init_page_text = _grid_updates(init_tasks, init_models, "All", 0)

    with gr.Blocks(title="Evaluation Results") as demo:
        gr.Markdown(f"# Simulation Evaluation Results\n`{root}`")
        stats_bar = gr.Markdown(_stats_markdown(init_d))

        with gr.Row():
            task_dd = gr.Dropdown(
                choices=init_tasks,
                value=init_tasks,
                multiselect=True,
                label="Tasks",
                filterable=True,
            )
            model_dd = gr.Dropdown(
                choices=init_models,
                value=init_models,
                multiselect=True,
                label="Models",
                filterable=True,
            )
            select_all_btn = gr.Button("Select All", scale=0, variant="secondary")
            refresh_btn = gr.Button("Refresh", scale=0, variant="secondary")

        # Load documentation markdown once for the Documentation tab.
        _doc_path = Path(__file__).resolve().parent.parent.parent / "tutorials" / "STATISTICAL_COMPARISON.md"
        _doc_content = _doc_path.read_text() if _doc_path.exists() else "*Documentation not found.*"

        with gr.Tabs():
            with gr.Tab("Summary"):
                sum_df = gr.Dataframe(_summary_df(init_d), interactive=False)
                csv_btn = gr.DownloadButton("Download CSV", scale=0)

            with gr.Tab("Model Comparison"):
                comparison_meta = gr.Markdown(_comparison_meta_text(init_comp_warning))
                gr.Markdown(
                    "*Beta posterior distributions with CLD letters from sequential statistical testing "
                    "(Bonferroni-corrected at global false positive rate = 0.05 "
                    "for each column). Horizontal lines show posterior means; dots show empirical means. "
                    "Shared CLD letter = not significantly different. "
                    "`successes / available (budgeted)` annotations are shown below each violin plot. "
                    "See **How Statistical Comparisons Work** in the **Documentation** tab for details.*"
                )
                bar_overlay_cb = gr.Checkbox(
                    label="Show bar overlay",
                    value=False,
                    container=False,
                )
                comparison_plot = gr.Plot(init_comp_fig)

            with gr.Tab("Spider"):
                spider_plot = gr.Plot(spider_chart(init_d))

            with gr.Tab("Episode Recordings"):
                with gr.Row():
                    rec_outcome = gr.Radio(["All", "Success", "Failure"], value="All", label="Outcome")
                    episode_filter_box = gr.Textbox(
                        label="Episode IDs",
                        placeholder="e.g. 0, 3, 12",
                        value="",
                        scale=1,
                    )
                with gr.Row():
                    prev_btn = gr.Button("< Prev", scale=0)
                    page_label = gr.Markdown(init_page_text)
                    next_btn = gr.Button("Next >", scale=0)
                page_state = gr.State(init_page)
                vid_slots = []
                link_slots = []
                for row_i in range(3):
                    with gr.Row():
                        for col_i in range(3):
                            idx = row_i * 3 + col_i
                            v = init_vids[idx]
                            lk = init_links[idx]
                            with gr.Column():
                                vid_slots.append(
                                    gr.Video(
                                        value=(v.get("value") if isinstance(v, dict) else None),
                                        visible=(v.get("visible", False) if isinstance(v, dict) else False),
                                        label=(v.get("label", "") if isinstance(v, dict) else ""),
                                        autoplay=True,
                                    )
                                )
                                link_slots.append(
                                    gr.Markdown(
                                        value=(lk.get("value", "") if isinstance(lk, dict) else ""),
                                        visible=(lk.get("visible", False) if isinstance(lk, dict) else False),
                                    )
                                )

            with gr.Tab("Documentation"):
                gr.Markdown(_doc_content)

        # ---- Wire events ----

        grid_out = vid_slots + link_slots + [page_state, page_label]
        filter_inputs = [task_dd, model_dd, bar_overlay_cb, rec_outcome, episode_filter_box]
        filter_out = [
            stats_bar,
            sum_df,
            comparison_meta,
            comparison_plot,
            spider_plot,
        ] + grid_out

        task_dd.change(on_filter, filter_inputs, filter_out)
        model_dd.change(on_filter, filter_inputs, filter_out)
        bar_overlay_cb.change(on_filter, filter_inputs, filter_out)

        select_all_btn.click(
            on_select_all,
            inputs=[bar_overlay_cb, rec_outcome, episode_filter_box],
            outputs=[task_dd, model_dd] + filter_out,
        )

        rec_filter_inputs = [task_dd, model_dd, rec_outcome, page_state, episode_filter_box]
        rec_outcome.change(on_rec_filter, rec_filter_inputs, grid_out)
        episode_filter_box.submit(on_rec_filter, rec_filter_inputs, grid_out)
        prev_btn.click(
            lambda *a: on_page(*a, direction=-1),
            [task_dd, model_dd, rec_outcome, page_state, episode_filter_box],
            grid_out,
        )
        next_btn.click(
            lambda *a: on_page(*a, direction=1),
            [task_dd, model_dd, rec_outcome, page_state, episode_filter_box],
            grid_out,
        )

        csv_btn.click(on_csv_download, outputs=csv_btn)

        refresh_btn.click(
            on_refresh,
            inputs=[bar_overlay_cb, rec_outcome, episode_filter_box],
            outputs=[task_dd, model_dd] + filter_out,
        )

    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def launch(root: Path, port: int = 8505):
    demo = create_app(root)
    demo.launch(server_name="0.0.0.0", server_port=port, allowed_paths=[str(root)])


def main():
    parser = argparse.ArgumentParser(description="Gradio evaluation dashboard.")
    parser.add_argument("rollout_dir", type=Path, nargs="?", default=Path("rollouts"))
    parser.add_argument("--port", type=int, default=8505)
    args = parser.parse_args()

    root = args.rollout_dir.resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        return 1

    launch(root, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
