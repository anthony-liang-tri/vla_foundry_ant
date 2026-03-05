"""Interactive curses TUI for exploring stats comparison results.

Launched automatically by compare_stats.py when stdout is a TTY (unless
``--no-interactive`` is passed).  Follows the curses patterns established
by :class:`DemoResultsViewer` in ``gather_results.py``.
"""

import contextlib
import curses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from vla_foundry.tri.utils.compare_stats import (
        COUNT_FIELDS,
        TensorInfo,
        extract_filter_axes,
        parse_tensor_name,
    )
except ImportError:
    from compare_stats import (
        COUNT_FIELDS,
        TensorInfo,
        extract_filter_axes,
        parse_tensor_name,
    )

# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

FILTER_CONFIG_PATH = Path("~/.config/vla_foundry/compare_stats_filters.json").expanduser()


@dataclass
class FilterState:
    """Persistent filter selections (empty set = show all)."""

    group: set[str] = field(default_factory=set)
    category: set[str] = field(default_factory=set)
    representation: set[str] = field(default_factory=set)
    limb: set[str] = field(default_factory=set)
    robot: set[str] = field(default_factory=set)

    sort_by: str = "max_pct"  # max_pct | avg_pct | max_abs | name
    sort_ascending: bool = False

    show_count_fields: bool = False


# Axis display order (used by filter panel columns)
AXIS_ORDER = ["group", "category", "representation", "limb", "robot"]
AXIS_LABELS = {
    "group": "Group",
    "category": "Category",
    "representation": "Repr",
    "limb": "Limb",
    "robot": "Robot",
}

# Unicode bar characters for diff magnitude visualization
BAR_CHARS = " ▏▎▍▌▋▊▉█"


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


class StatsComparisonViewer:
    """Curses-based interactive viewer for stats comparison data."""

    def __init__(
        self,
        our_path: str,
        ref_path: str,
        our_stats: dict,
        ref_stats: dict,
        comparison: dict,
    ):
        self.our_path = our_path
        self.ref_path = ref_path
        self.our_stats = our_stats
        self.ref_stats = ref_stats
        self.comparison = comparison

        # Parse tensor names
        all_tensors = sorted(set(our_stats.keys()) | set(ref_stats.keys()))
        self.tensor_infos: dict[str, TensorInfo] = {name: parse_tensor_name(name) for name in all_tensors}

        # Available filter values
        self.available_axes = extract_filter_axes(all_tensors)

        # Build summaries
        self.tensor_summary = self._build_tensor_summary()
        self.count_diffs = self._build_count_diffs()

        # Filter / sort state
        self.filters = FilterState()
        self._load_filters()

        # Navigation
        self.current_row = 0
        self.scroll_offset = 0

        # View mode: overview | detail | filter | side_by_side | help | info
        self.view_mode = "overview"
        self.selected_tensor: str | None = None
        self.detail_scroll = 0
        self.info_scroll = 0
        self._info_lines: list[str] = []  # built lazily on first info view

        # Filter panel state
        self.active_filter_axis = 0
        self.filter_cursor: dict[str, int] = {axis: 0 for axis in AXIS_ORDER}

        # Search
        self.search_term = ""

        # Filtered view
        self.filtered_summary: list[dict] = []
        self._apply_filters()

    # ------------------------------------------------------------------
    # Data building
    # ------------------------------------------------------------------

    def _build_tensor_summary(self) -> list[dict]:
        """Build the per-tensor summary list (mirrors print_report logic)."""
        summaries = []
        for tensor in sorted(self.comparison.get("common_tensors", [])):
            if tensor not in self.comparison.get("field_differences", {}):
                continue
            diffs = [d for d in self.comparison["field_differences"][tensor] if d["field"] not in COUNT_FIELDS]
            if not diffs:
                continue
            worst = max(diffs, key=lambda d: d["norm_diff"])
            summaries.append(
                {
                    "tensor": tensor,
                    "n": len(diffs),
                    "avg_pct": float(np.mean([d["norm_diff"] for d in diffs])),
                    "max_pct": worst["norm_diff"],
                    "max_abs": worst["max_abs_diff"],
                    "worst_field": worst["field"],
                }
            )
        summaries.sort(key=lambda x: x["max_pct"], reverse=True)
        return summaries

    def _build_count_diffs(self) -> list[dict]:
        """Build count-field diff list."""
        count_diffs = []
        for tensor, field_diffs in self.comparison.get("field_differences", {}).items():
            for d in field_diffs:
                if d["field"] in COUNT_FIELDS:
                    count_diffs.append({"tensor": tensor, **d})
        count_diffs.sort(key=lambda x: x["norm_diff"], reverse=True)
        return count_diffs

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _tensor_passes_filter(self, tensor_name: str) -> bool:
        """Check if a tensor passes all active filters."""
        info = self.tensor_infos.get(tensor_name)
        if info is None:
            return True

        for axis in AXIS_ORDER:
            active = getattr(self.filters, axis)
            if not active:
                continue  # empty = show all
            tensor_val = getattr(info, axis)
            if not tensor_val:
                continue  # tensor lacks this axis -> passes
            if tensor_val not in active:
                return False
        return True

    def _apply_filters(self):
        """Recompute filtered_summary from tensor_summary + filters + search + sort."""
        items = [s for s in self.tensor_summary if self._tensor_passes_filter(s["tensor"])]

        if self.filters.show_count_fields:
            for cd in self.count_diffs:
                if self._tensor_passes_filter(cd["tensor"]):
                    items.append(
                        {
                            "tensor": cd["tensor"],
                            "n": 1,
                            "avg_pct": cd["norm_diff"],
                            "max_pct": cd["norm_diff"],
                            "max_abs": cd["max_abs_diff"],
                            "worst_field": cd["field"],
                        }
                    )

        # Search filter
        if self.search_term:
            term = self.search_term.lower()
            items = [s for s in items if term in s["tensor"].lower()]

        # Sort
        sort_key_map = {
            "max_pct": lambda x: x["max_pct"],
            "avg_pct": lambda x: x["avg_pct"],
            "max_abs": lambda x: x["max_abs"],
            "name": lambda x: x["tensor"],
        }
        key_fn = sort_key_map.get(self.filters.sort_by, sort_key_map["max_pct"])
        items.sort(key=key_fn, reverse=not self.filters.sort_ascending)

        self.filtered_summary = items

        # Clamp navigation
        if self.current_row >= len(self.filtered_summary):
            self.current_row = max(0, len(self.filtered_summary) - 1)
        if self.scroll_offset > self.current_row:
            self.scroll_offset = self.current_row

    # ------------------------------------------------------------------
    # Filter persistence
    # ------------------------------------------------------------------

    def _save_filters(self):
        """Save filter state to ~/.config/vla_foundry/compare_stats_filters.json."""
        config = {axis: sorted(getattr(self.filters, axis)) for axis in AXIS_ORDER}
        config["sort_by"] = self.filters.sort_by
        config["sort_ascending"] = self.filters.sort_ascending
        config["show_count_fields"] = self.filters.show_count_fields
        try:
            FILTER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(FILTER_CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=2)
        except OSError:
            pass

    def _load_filters(self):
        """Load filter state if config exists."""
        if not FILTER_CONFIG_PATH.exists():
            return
        try:
            with open(FILTER_CONFIG_PATH) as f:
                config = json.load(f)
            for axis in AXIS_ORDER:
                setattr(self.filters, axis, set(config.get(axis, [])))
            self.filters.sort_by = config.get("sort_by", "max_pct")
            self.filters.sort_ascending = config.get("sort_ascending", False)
            self.filters.show_count_fields = config.get("show_count_fields", False)
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def safe_addstr(stdscr, y: int, x: int, text: str, attr: int = 0):
        """Write text to screen, silently ignoring curses boundary errors."""
        with contextlib.suppress(curses.error):
            stdscr.addstr(y, x, text, attr) if attr else stdscr.addstr(y, x, text)

    @staticmethod
    def _short_tensor(name: str, width: int = 40) -> str:
        if len(name) <= width:
            return name
        return "..." + name[-(width - 3) :]

    def _severity_color(self, pct: float) -> int:
        """Color pair number by diff severity (3=green, 4=yellow, 5=red)."""
        if pct < 1.0:
            return 3
        elif pct < 5.0:
            return 4
        return 5

    @staticmethod
    def _diff_bar(pct: float, width: int = 10) -> str:
        """Render a Unicode bar proportional to *pct* (clamped 0-100)."""
        pct = max(0.0, min(pct, 100.0))
        filled = pct / 100.0 * width
        full_blocks = int(filled)
        remainder = filled - full_blocks
        idx = int(remainder * (len(BAR_CHARS) - 1))
        bar = BAR_CHARS[-1] * full_blocks
        if full_blocks < width:
            bar += BAR_CHARS[idx]
            bar += " " * (width - full_blocks - 1)
        return bar[:width]

    def _format_active_filters(self) -> str:
        """One-line summary of active filters."""
        parts = []
        for axis in AXIS_ORDER:
            vals = getattr(self.filters, axis)
            if vals:
                label = AXIS_LABELS[axis]
                parts.append(f"{label}={','.join(sorted(vals))}")
        return "  ".join(parts) if parts else "(none)"

    def _ordered_tensors(self) -> list[str]:
        """Tensor names in the filtered_summary order."""
        return [s["tensor"] for s in self.filtered_summary]

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------

    def _init_colors(self):
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)  # selected row
        curses.init_pair(2, curses.COLOR_CYAN, -1)  # header
        curses.init_pair(3, curses.COLOR_GREEN, -1)  # good (<1%)
        curses.init_pair(4, curses.COLOR_YELLOW, -1)  # medium (1-5%)
        curses.init_pair(5, curses.COLOR_RED, -1)  # bad (>5%)
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # info
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_GREEN)  # filter active
        curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_RED)  # filter title
        curses.init_pair(9, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # search hl

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, stdscr):
        """Curses main loop — passed to ``curses.wrapper``."""
        curses.curs_set(0)
        curses.use_default_colors()
        self._init_colors()

        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()

            if height < 5 or width < 40:
                self.safe_addstr(stdscr, 0, 0, "Terminal too small")
                stdscr.refresh()
                stdscr.getch()
                continue

            if self.view_mode == "overview":
                self._draw_overview(stdscr, height, width)
            elif self.view_mode == "detail":
                self._draw_detail(stdscr, height, width)
            elif self.view_mode == "filter":
                self._draw_filter_panel(stdscr, height, width)
            elif self.view_mode == "side_by_side":
                self._draw_side_by_side(stdscr, height, width)
            elif self.view_mode == "help":
                self._draw_help(stdscr, height, width)
            elif self.view_mode == "info":
                self._draw_info(stdscr, height, width)

            stdscr.refresh()
            key = stdscr.getch()

            if not self._handle_input(key, stdscr, height, width):
                break

    # ==================================================================
    # OVERVIEW
    # ==================================================================

    def _draw_overview(self, stdscr, height: int, width: int):
        # --- header (lines 0-3) ---
        title = " STATS COMPARISON (interactive) "
        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        self.safe_addstr(stdscr, 0, 0, title.center(width)[:width])
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

        paths = f" Ours: {self.our_path}  |  Ref: {self.ref_path}"
        self.safe_addstr(stdscr, 1, 0, paths[:width], curses.color_pair(6))

        n_show = len(self.filtered_summary)
        n_total = len(self.tensor_summary)
        info_line = f" Showing {n_show}/{n_total} tensors"
        active = self._format_active_filters()
        if active != "(none)":
            info_line += f"  |  Filters: {active}"
        if self.search_term:
            info_line += f'  |  Search: "{self.search_term}"'
        self.safe_addstr(stdscr, 2, 0, info_line[:width])

        self.safe_addstr(stdscr, 3, 0, "\u2500" * width)

        # --- table header (line 4) ---
        TW = max(30, width - 62)
        sort_ind = {k: " " for k in ("name", "avg_pct", "max_pct", "max_abs")}
        arrow = "\u25b2" if self.filters.sort_ascending else "\u25bc"
        sort_ind[self.filters.sort_by] = arrow

        hdr = (
            f"{'Tensor':<{TW}s} {'#':>3s} "
            f"{'Worst field':<20s} "
            f"{'MaxAbs':>10s}{sort_ind['max_abs']} "
            f"{'Avg%':>7s}{sort_ind['avg_pct']} "
            f"{'Max%':>7s}{sort_ind['max_pct']}"
        )
        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        self.safe_addstr(stdscr, 4, 0, hdr[:width])
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

        # --- rows ---
        list_height = height - 7  # header=5, footer=2
        items = self.filtered_summary
        for idx in range(max(0, list_height)):
            row_idx = idx + self.scroll_offset
            if row_idx >= len(items):
                break
            item = items[row_idx]
            y = 5 + idx

            t = self._short_tensor(item["tensor"], TW)
            f_name = item["worst_field"]
            if len(f_name) > 20:
                f_name = f_name[-20:]
            row_text = (
                f"{t:<{TW}s} {item['n']:>3d} "
                f"{f_name:<20s} "
                f"{item['max_abs']:>10.6f}  "
                f"{item['avg_pct']:>6.2f}% "
                f"{item['max_pct']:>6.2f}%"
            )

            if row_idx == self.current_row:
                self.safe_addstr(stdscr, y, 0, row_text.ljust(width)[:width], curses.color_pair(1) | curses.A_BOLD)
            else:
                color = curses.color_pair(self._severity_color(item["max_pct"]))
                self.safe_addstr(stdscr, y, 0, row_text[:width], color)

        # --- footer ---
        footer_y = height - 2
        self.safe_addstr(stdscr, footer_y, 0, "\u2500" * width)
        keys = " j/k:nav  Enter:detail  f:filter  s:side-by-side  /:search  o/O:sort  i:info  ?:help  q:quit "
        self.safe_addstr(stdscr, height - 1, 0, keys[:width], curses.A_DIM)

    # ==================================================================
    # DETAIL VIEW
    # ==================================================================

    def _format_sample_count_line(self, tensor: str, width: int) -> str:
        """Build a human-readable sample count summary for *tensor*."""
        sc = self.comparison.get("sample_counts", {}).get(tensor)
        if not sc:
            return ""
        oc, rc = sc["our_count"], sc["ref_count"]
        if oc == rc:
            return f" Samples at anchor: {oc:,d} (identical)"
        diff = oc - rc
        pct = abs(diff) / rc * 100 if rc > 0 else float("inf")
        if diff > 0:
            who = f"ours has {pct:.1f}% MORE data (+{diff:,d})"
        else:
            who = f"ours has {pct:.1f}% LESS data ({diff:,d})"
        return f" Samples at anchor: ours {oc:,d}  ref {rc:,d}  \u2014 {who}"

    def _draw_detail(self, stdscr, height: int, width: int):
        tensor = self.selected_tensor
        if tensor is None:
            self.view_mode = "overview"
            return

        # Header
        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        self.safe_addstr(stdscr, 0, 0, f" DETAIL: {tensor} ".center(width)[:width])
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

        # Position in list + sample counts
        tensors = self._ordered_tensors()
        pos = tensors.index(tensor) + 1 if tensor in tensors else 0
        self.safe_addstr(stdscr, 1, 0, f" Tensor {pos}/{len(tensors)}", curses.color_pair(6))

        sc_line = self._format_sample_count_line(tensor, width)
        if sc_line:
            sc = self.comparison["sample_counts"][tensor]
            color = curses.color_pair(4) if sc["our_count"] != sc["ref_count"] else 0
            self.safe_addstr(stdscr, 2, 0, sc_line[:width], color)
            data_y = 3
        else:
            data_y = 2

        self.safe_addstr(stdscr, data_y, 0, "\u2500" * width)

        # Column header
        hdr = f"{'Field':<30s} {'Range':>19s} {'MaxAbs':>12s} {'MeanAbs':>12s} {'%Range':>8s}  {'Bar':<12s}"
        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        self.safe_addstr(stdscr, data_y + 1, 0, hdr[:width])
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

        # Field rows (skip count fields — sample info is in the header)
        diffs = [
            d for d in self.comparison.get("field_differences", {}).get(tensor, []) if d["field"] not in COUNT_FIELDS
        ]
        first_row_y = data_y + 2
        list_height = height - first_row_y - 2  # footer = 2 lines
        for idx in range(max(0, list_height)):
            row_idx = idx + self.detail_scroll
            if row_idx >= len(diffs):
                break
            d = diffs[row_idx]
            y = first_row_y + idx
            f_name = d["field"]

            if len(f_name) > 30:
                f_name = "..." + f_name[-27:]

            fmin = d.get("field_min", 0.0)
            fmax = d.get("field_max", 0.0)
            range_str = f"[{fmin:.1f}, {fmax:.1f}]"

            bar = self._diff_bar(d["norm_diff"])
            row = (
                f"{f_name:<30s} {range_str:>19s} {d['max_abs_diff']:>12.6f} {d['mean_abs_diff']:>12.6f} "
                f"{d['norm_diff']:>7.2f}%  {bar}"
            )
            color = curses.color_pair(self._severity_color(d["norm_diff"]))
            self.safe_addstr(stdscr, y, 0, row[:width], color)

        footer_y = height - 2
        self.safe_addstr(stdscr, footer_y, 0, "\u2500" * width)
        keys = " j/k:scroll  Left/Right:prev/next tensor  s:side-by-side  Esc:back "
        self.safe_addstr(stdscr, height - 1, 0, keys[:width], curses.A_DIM)

    # ==================================================================
    # FILTER PANEL
    # ==================================================================

    def _draw_filter_panel(self, stdscr, height: int, width: int):
        stdscr.attron(curses.color_pair(8) | curses.A_BOLD)
        self.safe_addstr(stdscr, 0, 0, " FILTER PANEL ".center(width)[:width])
        stdscr.attroff(curses.color_pair(8) | curses.A_BOLD)

        self.safe_addstr(
            stdscr,
            1,
            0,
            " Left/Right:axis  j/k:navigate  Space:toggle  a:all  n:none  Enter/Esc:apply"[:width],
            curses.A_DIM,
        )
        self.safe_addstr(stdscr, 2, 0, "\u2500" * width)

        # Compute column widths
        n_axes = len(AXIS_ORDER)
        col_w = max(15, (width - 2) // n_axes)

        # Draw column headers
        for ci, axis in enumerate(AXIS_ORDER):
            x = ci * col_w
            label = AXIS_LABELS[axis]
            vals = sorted(self.available_axes.get(axis, set()))
            count = len(vals)
            selected = getattr(self.filters, axis)
            sel_count = len(selected)
            header_text = f" {label} ({sel_count}/{count})"

            if ci == self.active_filter_axis:
                attr = curses.color_pair(1) | curses.A_BOLD
            else:
                attr = curses.color_pair(2) | curses.A_BOLD
            self.safe_addstr(stdscr, 3, x, header_text.ljust(col_w)[:col_w], attr)

        # Draw values
        for ci, axis in enumerate(AXIS_ORDER):
            x = ci * col_w
            vals = sorted(self.available_axes.get(axis, set()))
            selected = getattr(self.filters, axis)
            cursor_pos = self.filter_cursor.get(axis, 0)

            for vi, val in enumerate(vals):
                y = 4 + vi
                if y >= height - 2:
                    break
                checked = "[x]" if val in selected else "[ ]"
                text = f" {checked} {val}"

                is_active_col = ci == self.active_filter_axis
                is_cursor = is_active_col and vi == cursor_pos

                if is_cursor:
                    attr = curses.color_pair(1) | curses.A_BOLD
                elif val in selected:
                    attr = curses.color_pair(3)
                else:
                    attr = curses.A_NORMAL

                self.safe_addstr(stdscr, y, x, text.ljust(col_w)[:col_w], attr)

        footer_y = height - 1
        self.safe_addstr(stdscr, footer_y, 0, " Space:toggle  a:all  n:none  Enter/Esc:apply "[:width], curses.A_DIM)

    # ==================================================================
    # SIDE-BY-SIDE VIEW
    # ==================================================================

    def _draw_side_by_side(self, stdscr, height: int, width: int):
        tensor = self.selected_tensor
        if tensor is None:
            self.view_mode = "overview"
            return

        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        self.safe_addstr(stdscr, 0, 0, f" SIDE-BY-SIDE: {tensor} ".center(width)[:width])
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

        tensors = self._ordered_tensors()
        pos = tensors.index(tensor) + 1 if tensor in tensors else 0
        self.safe_addstr(stdscr, 1, 0, f" Tensor {pos}/{len(tensors)}", curses.color_pair(6))
        self.safe_addstr(stdscr, 2, 0, "\u2500" * width)

        our_tensor = self.our_stats.get(tensor, {})
        ref_tensor = self.ref_stats.get(tensor, {})
        all_fields = sorted(set(our_tensor.keys()) | set(ref_tensor.keys()))

        # Column header
        half = max(20, (width - 32) // 2)
        hdr = f"{'Field':<30s} {'Ours':>{half}s} {'Ref':>{half}s}"
        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        self.safe_addstr(stdscr, 3, 0, hdr[:width])
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

        list_height = height - 6
        for idx in range(max(0, list_height)):
            row_idx = idx + self.detail_scroll
            if row_idx >= len(all_fields):
                break
            fld = all_fields[row_idx]
            y = 4 + idx

            our_val = our_tensor.get(fld)
            ref_val = ref_tensor.get(fld)

            our_str = self._format_value(our_val, half)
            ref_str = self._format_value(ref_val, half)
            row = f"{fld:<30s} {our_str:>{half}s} {ref_str:>{half}s}"
            self.safe_addstr(stdscr, y, 0, row[:width])

        footer_y = height - 2
        self.safe_addstr(stdscr, footer_y, 0, "\u2500" * width)
        keys = " j/k:scroll  Left/Right:prev/next tensor  Esc:back "
        self.safe_addstr(stdscr, height - 1, 0, keys[:width], curses.A_DIM)

    @staticmethod
    def _format_value(val, max_width: int = 30) -> str:
        """Format a stat value for side-by-side display."""
        if val is None:
            return "None"
        if isinstance(val, (int, float)):
            return f"{val:.6g}"
        if isinstance(val, list):
            if len(val) == 0:
                return "[]"
            # Flat list of numbers
            if isinstance(val[0], (int, float)):
                if len(val) <= 4:
                    return "[" + ", ".join(f"{v:.4g}" for v in val) + "]"
                return f"[{val[0]:.4g}..{val[-1]:.4g}] ({len(val)})"
            # Nested list
            return f"[{len(val)} x ...]"
        s = str(val)
        if len(s) > max_width:
            return s[: max_width - 3] + "..."
        return s

    # ==================================================================
    # HELP OVERLAY
    # ==================================================================

    def _draw_help(self, stdscr, height: int, width: int):
        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        self.safe_addstr(stdscr, 0, 0, " KEY BINDINGS ".center(width)[:width])
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

        help_lines = [
            "",
            "OVERVIEW",
            "  j / k / Up / Down   Navigate tensor list",
            "  PgUp / PgDn         Page up / down",
            "  Enter               Open tensor detail view",
            "  f                   Open filter panel",
            "  s                   Side-by-side view for selected tensor",
            "  i                   Composition info overlay",
            "  /                   Search tensors by name",
            "  o                   Cycle sort column (max%, avg%, maxAbs, name)",
            "  O (shift-o)         Toggle sort direction",
            "  c                   Toggle count fields in overview",
            "  S (shift-s)         Save current filters to disk",
            "  L (shift-l)         Load saved filters from disk",
            "  r                   Reset all filters",
            "  e                   Export: print filtered report to stdout",
            "  ?                   This help screen",
            "  q / Esc             Quit (or back from sub-view)",
            "",
            "FILTER PANEL",
            "  Left / Right        Switch axis column",
            "  j / k / Up / Down   Navigate values",
            "  Space               Toggle value on/off",
            "  a                   Select all values in column",
            "  n                   Select none in column",
            "  Enter / Esc         Apply and return to overview",
            "",
            "DETAIL / SIDE-BY-SIDE",
            "  j / k / Up / Down   Scroll field list",
            "  Left / Right        Previous / next tensor",
            "  s                   Switch to side-by-side (from detail)",
            "  Esc                 Back to overview",
            "",
            "Press any key to close this help screen.",
        ]

        for i, line in enumerate(help_lines):
            if i + 1 >= height:
                break
            self.safe_addstr(stdscr, 1 + i, 0, line[:width])

    # ==================================================================
    # INFO OVERLAY (scrollable)
    # ==================================================================

    def _build_info_lines(self) -> list[str]:
        """Build the full info content (called once, cached)."""
        lines: list[str] = []
        lines.append(f"Ours: {self.our_path}")
        lines.append(f"Ref:  {self.ref_path}")
        lines.append("")

        # --- Anchors ---
        anchor = self.comparison.get("anchor_info", {})
        lines.append(f"Our anchor: {anchor.get('our_anchor')}  (source: {anchor.get('our_source', '?')})")
        lines.append(f"Ref anchor: {anchor.get('ref_anchor')}  (source: {anchor.get('ref_source', '?')})")
        lines.append("")

        # --- Tensor counts ---
        common = self.comparison.get("common_tensors", set())
        only_ours = self.comparison.get("only_in_ours", set())
        only_ref = self.comparison.get("only_in_ref", set())
        lines.append(f"Common tensors: {len(common)}")
        lines.append(f"Only in ours:   {len(only_ours)}")
        lines.append(f"Only in ref:    {len(only_ref)}")
        lines.append("")

        # --- Timestep window ---
        ts_info = self.comparison.get("timestep_info", {})
        if ts_info:
            ex = next(iter(ts_info.values()))
            our_past = ex["our_anchor"]
            our_future = ex["our_len"] - ex["our_anchor"] - 1
            ref_past = ex["ref_anchor"]
            ref_future = ex["ref_len"] - ex["ref_anchor"] - 1
            overlap = min(our_past, ref_past) + min(our_future, ref_future) + 1
            lines.append(f"Our window:  {ex['our_len']} timesteps (past={our_past}, future={our_future})")
            lines.append(f"Ref window:  {ex['ref_len']} timesteps (past={ref_past}, future={ref_future})")
            lines.append(f"Overlap:     {overlap} timesteps")
            lines.append("")

        # --- Sample counts summary ---
        sc = self.comparison.get("sample_counts", {})
        if sc:
            from collections import Counter

            our_counts = {t: v["our_count"] for t, v in sc.items()}
            ref_counts = {t: v["ref_count"] for t, v in sc.items()}
            our_vals = sorted(set(our_counts.values()))
            ref_vals = sorted(set(ref_counts.values()))
            our_mode = Counter(our_counts.values()).most_common(1)[0][0]
            ref_mode = Counter(ref_counts.values()).most_common(1)[0][0]

            lines.append("SAMPLE COUNTS (at anchor timestep):")
            lines.append(f"  Ours: {our_mode:>14,d} samples")
            lines.append(f"  Ref:  {ref_mode:>14,d} samples")

            diff = our_mode - ref_mode
            if diff == 0:
                lines.append("  \u2192 Identical sample counts")
            elif ref_mode > 0:
                pct = abs(diff) / ref_mode * 100
                if diff > 0:
                    lines.append(f"  \u2192 OURS has {pct:.1f}% MORE data (+{diff:,d} samples)")
                else:
                    lines.append(f"  \u2192 OURS has {pct:.1f}% LESS data ({diff:,d} samples)")
            lines.append("")

            # Variation across tensors
            if len(our_vals) > 1 or len(ref_vals) > 1:
                lines.append("!! Sample counts VARY across tensors:")
                lines.append(f"   Ours range: {min(our_vals):,d} \u2013 {max(our_vals):,d}")
                lines.append(f"   Ref  range: {min(ref_vals):,d} \u2013 {max(ref_vals):,d}")
                outliers = []
                for t in sorted(sc.keys()):
                    oc, rc = our_counts[t], ref_counts[t]
                    if oc != our_mode or rc != ref_mode:
                        outliers.append((t, oc, rc))
                if outliers:
                    lines.append(f"   Typical: ours={our_mode:,d}  ref={ref_mode:,d}")
                    lines.append("   Outlier tensors:")
                    for t, oc, rc in outliers:
                        d = oc - rc
                        sign = "+" if d > 0 else ""
                        lines.append(f"     {t}:  ours={oc:,d}  ref={rc:,d}  ({sign}{d:,d})")
                lines.append("")

        # --- Missing tensors (only in ours) ---
        if only_ours:
            lines.append(f"TENSORS ONLY IN OURS ({len(only_ours)}):")
            for t in sorted(only_ours):
                info = self.tensor_infos.get(t)
                suffix = ""
                if info and info.group:
                    parts = [info.group]
                    if info.category:
                        parts.append(info.category)
                    if info.limb:
                        parts.append(info.limb)
                    suffix = f"  [{'/'.join(parts)}]"
                lines.append(f"  {t}{suffix}")
            lines.append("")

        # --- Missing tensors (only in ref) ---
        if only_ref:
            lines.append(f"TENSORS ONLY IN REF ({len(only_ref)}):")
            for t in sorted(only_ref):
                info = self.tensor_infos.get(t)
                suffix = ""
                if info and info.group:
                    parts = [info.group]
                    if info.category:
                        parts.append(info.category)
                    if info.limb:
                        parts.append(info.limb)
                    suffix = f"  [{'/'.join(parts)}]"
                lines.append(f"  {t}{suffix}")
            lines.append("")

        # --- Per-tensor sample count table ---
        if sc and len(sc) > 1:
            lines.append(f"PER-TENSOR SAMPLE COUNTS ({len(sc)} tensors):")
            TW = 50
            lines.append(f"  {'Tensor':<{TW}s} {'Ours':>10s} {'Ref':>10s} {'Diff':>10s}")
            lines.append(f"  {'-' * TW} {'-' * 10} {'-' * 10} {'-' * 10}")
            for t in sorted(sc.keys()):
                v = sc[t]
                oc, rc = v["our_count"], v["ref_count"]
                d = oc - rc
                tn = t if len(t) <= TW else "..." + t[-(TW - 3) :]
                sign = "+" if d > 0 else ""
                lines.append(f"  {tn:<{TW}s} {oc:>10,d} {rc:>10,d} {sign}{d:>9,d}")
            lines.append("")

        # --- None values ---
        none_ours = self.comparison.get("none_values", {}).get("our", {})
        none_ref = self.comparison.get("none_values", {}).get("ref", {})
        if none_ours or none_ref:
            lines.append("NONE VALUES:")
            if none_ours:
                lines.append(f"  In ours ({len(none_ours)} tensors):")
                for t in sorted(none_ours.keys()):
                    fields = ", ".join(sorted(none_ours[t]))
                    lines.append(f"    {t}: {fields}")
            if none_ref:
                lines.append(f"  In ref ({len(none_ref)} tensors):")
                for t in sorted(none_ref.keys()):
                    fields = ", ".join(sorted(none_ref[t]))
                    lines.append(f"    {t}: {fields}")
            lines.append("")

        lines.append("j/k:scroll  Esc:back")
        return lines

    def _draw_info(self, stdscr, height: int, width: int):
        if not self._info_lines:
            self._info_lines = self._build_info_lines()

        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        title = f" COMPOSITION INFO ({self.info_scroll + 1}/{len(self._info_lines)}) "
        self.safe_addstr(stdscr, 0, 0, title.center(width)[:width])
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

        view_height = height - 2
        for i in range(view_height):
            line_idx = i + self.info_scroll
            if line_idx >= len(self._info_lines):
                break
            line = self._info_lines[line_idx]
            # Color-code section headers
            if line.startswith("TENSORS ONLY") or line.startswith("PER-TENSOR") or line.startswith("NONE VALUES"):
                attr = curses.color_pair(4) | curses.A_BOLD
            elif line.startswith("!!"):
                attr = curses.color_pair(5) | curses.A_BOLD
            else:
                attr = 0
            self.safe_addstr(stdscr, 1 + i, 0, line[:width], attr)

        footer = " j/k:scroll  PgUp/PgDn:page  Esc/q:back "
        self.safe_addstr(stdscr, height - 1, 0, footer[:width], curses.A_DIM)

    # ==================================================================
    # SEARCH
    # ==================================================================

    def _show_search(self, stdscr, height: int, width: int):
        """Prompt for search term."""
        prompt = "Search: "
        y = height // 2
        self.safe_addstr(stdscr, y, 0, " " * width)
        self.safe_addstr(stdscr, y, 2, prompt, curses.color_pair(2) | curses.A_BOLD)
        stdscr.refresh()

        curses.echo()
        curses.curs_set(1)
        try:
            search_bytes = stdscr.getstr(y, 2 + len(prompt), 60)
            self.search_term = search_bytes.decode("utf-8", errors="replace").strip()
        except Exception:
            self.search_term = ""
        curses.noecho()
        curses.curs_set(0)

        self.current_row = 0
        self.scroll_offset = 0
        self._apply_filters()

    # ==================================================================
    # EXPORT
    # ==================================================================

    def _export_filtered(self, stdscr):
        """Temporarily exit curses and print a filtered report to stdout."""
        curses.endwin()

        print(f"\n{'  FILTERED STATS COMPARISON  ':=^80}")
        print(f"Ours: {self.our_path}")
        print(f"Ref:  {self.ref_path}")
        print(f"Active filters: {self._format_active_filters()}")
        if self.search_term:
            print(f'Search: "{self.search_term}"')
        print(f"\nShowing {len(self.filtered_summary)}/{len(self.tensor_summary)} tensors\n")

        TW = 40
        print(f"{'Tensor':<{TW}s} {'#':>3s} {'Worst field':<22s} {'MaxAbs':>12s} {'Avg%':>7s} {'Max%':>7s}")
        print(f"{'-' * TW} {'-' * 3} {'-' * 22} {'-' * 12} {'-' * 7} {'-' * 7}")
        for item in self.filtered_summary:
            t = self._short_tensor(item["tensor"], TW)
            f = item["worst_field"][-22:] if len(item["worst_field"]) > 22 else item["worst_field"]
            print(
                f"{t:<{TW}s} {item['n']:>3d} {f:<22s} "
                f"{item['max_abs']:>12.6f} {item['avg_pct']:>6.2f}% {item['max_pct']:>6.2f}%"
            )
        print(f"\n{'':=^80}")
        print("\nPress Enter to return to TUI...", end="", flush=True)
        with contextlib.suppress(EOFError):
            input()

        # Re-init curses
        stdscr.refresh()

    # ==================================================================
    # INPUT HANDLING
    # ==================================================================

    def _handle_input(self, key: int, stdscr, height: int, width: int) -> bool:
        """Dispatch key press. Return False to quit."""
        if self.view_mode == "overview":
            return self._handle_overview_input(key, stdscr, height, width)
        elif self.view_mode == "detail":
            return self._handle_detail_input(key, stdscr, height, width)
        elif self.view_mode == "filter":
            return self._handle_filter_input(key, stdscr, height, width)
        elif self.view_mode == "side_by_side":
            return self._handle_side_by_side_input(key, stdscr, height, width)
        elif self.view_mode == "help":
            self.view_mode = "overview"
            return True
        elif self.view_mode == "info":
            return self._handle_info_input(key, stdscr, height, width)
        return True

    # --- Overview input ---

    def _handle_overview_input(self, key: int, stdscr, height: int, width: int) -> bool:
        n_items = len(self.filtered_summary)
        list_height = max(1, height - 7)

        if key in (ord("q"), ord("Q")) or key == 27:
            self._save_filters()
            return False
        elif key in (curses.KEY_DOWN, ord("j")):
            if self.current_row < n_items - 1:
                self.current_row += 1
                if self.current_row - self.scroll_offset >= list_height:
                    self.scroll_offset += 1
        elif key in (curses.KEY_UP, ord("k")):
            if self.current_row > 0:
                self.current_row -= 1
                if self.current_row < self.scroll_offset:
                    self.scroll_offset -= 1
        elif key == curses.KEY_PPAGE:
            self.current_row = max(0, self.current_row - list_height)
            self.scroll_offset = max(0, self.scroll_offset - list_height)
        elif key == curses.KEY_NPAGE:
            self.current_row = min(n_items - 1, self.current_row + list_height)
            max_off = max(0, n_items - list_height)
            self.scroll_offset = min(max_off, self.scroll_offset + list_height)
        elif key == curses.KEY_HOME:
            self.current_row = 0
            self.scroll_offset = 0
        elif key == curses.KEY_END:
            self.current_row = max(0, n_items - 1)
            self.scroll_offset = max(0, n_items - list_height)
        elif key in (curses.KEY_ENTER, ord("\n"), 10):
            if n_items > 0:
                self.selected_tensor = self.filtered_summary[self.current_row]["tensor"]
                self.detail_scroll = 0
                self.view_mode = "detail"
        elif key == ord("f"):
            self.view_mode = "filter"
        elif key == ord("s"):
            if n_items > 0:
                self.selected_tensor = self.filtered_summary[self.current_row]["tensor"]
                self.detail_scroll = 0
                self.view_mode = "side_by_side"
        elif key == ord("i"):
            self.info_scroll = 0
            self._info_lines = []  # rebuild fresh
            self.view_mode = "info"
        elif key == ord("?"):
            self.view_mode = "help"
        elif key == ord("/"):
            self._show_search(stdscr, height, width)
        elif key == ord("o"):
            # Cycle sort column
            columns = ["max_pct", "avg_pct", "max_abs", "name"]
            idx = columns.index(self.filters.sort_by) if self.filters.sort_by in columns else 0
            self.filters.sort_by = columns[(idx + 1) % len(columns)]
            self._apply_filters()
            self._save_filters()
        elif key == ord("O"):
            self.filters.sort_ascending = not self.filters.sort_ascending
            self._apply_filters()
            self._save_filters()
        elif key == ord("c"):
            self.filters.show_count_fields = not self.filters.show_count_fields
            self._apply_filters()
            self._save_filters()
        elif key == ord("S"):
            self._save_filters()
        elif key == ord("L"):
            self._load_filters()
            self._apply_filters()
        elif key == ord("r"):
            self.filters = FilterState()
            self.search_term = ""
            self.current_row = 0
            self.scroll_offset = 0
            self._apply_filters()
            self._save_filters()
        elif key == ord("e"):
            self._export_filtered(stdscr)

        return True

    # --- Detail input ---

    def _handle_detail_input(self, key: int, stdscr, height: int, width: int) -> bool:
        tensor = self.selected_tensor
        diffs = [
            d for d in self.comparison.get("field_differences", {}).get(tensor, []) if d["field"] not in COUNT_FIELDS
        ]
        list_height = max(1, height - 6)

        if key == 27 or key == ord("q"):  # Esc or q
            self.view_mode = "overview"
        elif key in (curses.KEY_DOWN, ord("j")):
            if self.detail_scroll < len(diffs) - 1:
                self.detail_scroll += 1
        elif key in (curses.KEY_UP, ord("k")):
            if self.detail_scroll > 0:
                self.detail_scroll -= 1
        elif key == curses.KEY_PPAGE:
            self.detail_scroll = max(0, self.detail_scroll - list_height)
        elif key == curses.KEY_NPAGE:
            self.detail_scroll = min(max(0, len(diffs) - 1), self.detail_scroll + list_height)
        elif key == curses.KEY_LEFT:
            self._navigate_tensor(-1)
        elif key == curses.KEY_RIGHT:
            self._navigate_tensor(1)
        elif key == ord("s"):
            self.detail_scroll = 0
            self.view_mode = "side_by_side"
        return True

    # --- Side-by-side input ---

    def _handle_side_by_side_input(self, key: int, stdscr, height: int, width: int) -> bool:
        tensor = self.selected_tensor
        our_tensor = self.our_stats.get(tensor, {})
        ref_tensor = self.ref_stats.get(tensor, {})
        n_fields = len(set(our_tensor.keys()) | set(ref_tensor.keys()))
        list_height = max(1, height - 6)

        if key == 27 or key == ord("q"):  # Esc or q
            self.view_mode = "overview"
        elif key in (curses.KEY_DOWN, ord("j")):
            if self.detail_scroll < n_fields - 1:
                self.detail_scroll += 1
        elif key in (curses.KEY_UP, ord("k")):
            if self.detail_scroll > 0:
                self.detail_scroll -= 1
        elif key == curses.KEY_PPAGE:
            self.detail_scroll = max(0, self.detail_scroll - list_height)
        elif key == curses.KEY_NPAGE:
            self.detail_scroll = min(max(0, n_fields - 1), self.detail_scroll + list_height)
        elif key == curses.KEY_LEFT:
            self._navigate_tensor(-1)
        elif key == curses.KEY_RIGHT:
            self._navigate_tensor(1)
        return True

    # --- Filter input ---

    def _handle_filter_input(self, key: int, stdscr, height: int, width: int) -> bool:
        axis = AXIS_ORDER[self.active_filter_axis]
        vals = sorted(self.available_axes.get(axis, set()))
        n_vals = len(vals)
        cursor = self.filter_cursor.get(axis, 0)

        if key in (27, curses.KEY_ENTER, ord("\n"), 10):  # Esc or Enter
            self.view_mode = "overview"
            self.current_row = 0
            self.scroll_offset = 0
            self._apply_filters()
            self._save_filters()
        elif key == curses.KEY_LEFT:
            self.active_filter_axis = (self.active_filter_axis - 1) % len(AXIS_ORDER)
        elif key == curses.KEY_RIGHT:
            self.active_filter_axis = (self.active_filter_axis + 1) % len(AXIS_ORDER)
        elif key in (curses.KEY_DOWN, ord("j")):
            if cursor < n_vals - 1:
                self.filter_cursor[axis] = cursor + 1
        elif key in (curses.KEY_UP, ord("k")):
            if cursor > 0:
                self.filter_cursor[axis] = cursor - 1
        elif key == ord(" "):
            if n_vals > 0 and cursor < n_vals:
                val = vals[cursor]
                selected = getattr(self.filters, axis)
                if val in selected:
                    selected.discard(val)
                else:
                    selected.add(val)
        elif key == ord("a"):
            setattr(self.filters, axis, set(vals))
        elif key == ord("n"):
            setattr(self.filters, axis, set())
        return True

    # --- Info input ---

    def _handle_info_input(self, key: int, stdscr, height: int, width: int) -> bool:
        n_lines = len(self._info_lines)
        view_height = max(1, height - 2)

        if key in (27, ord("q")):  # Esc or q
            self.view_mode = "overview"
            self.info_scroll = 0
            self._info_lines = []  # free memory, rebuild next time
        elif key in (curses.KEY_DOWN, ord("j")):
            if self.info_scroll < n_lines - 1:
                self.info_scroll += 1
        elif key in (curses.KEY_UP, ord("k")):
            if self.info_scroll > 0:
                self.info_scroll -= 1
        elif key == curses.KEY_PPAGE:
            self.info_scroll = max(0, self.info_scroll - view_height)
        elif key == curses.KEY_NPAGE:
            self.info_scroll = min(max(0, n_lines - 1), self.info_scroll + view_height)
        elif key == curses.KEY_HOME:
            self.info_scroll = 0
        elif key == curses.KEY_END:
            self.info_scroll = max(0, n_lines - view_height)
        return True

    # --- Navigation helpers ---

    def _navigate_tensor(self, direction: int):
        """Move to prev/next tensor in filtered list."""
        tensors = self._ordered_tensors()
        if not tensors:
            return
        try:
            idx = tensors.index(self.selected_tensor)
        except ValueError:
            idx = 0
        new_idx = max(0, min(len(tensors) - 1, idx + direction))
        self.selected_tensor = tensors[new_idx]
        self.detail_scroll = 0


if __name__ == "__main__":
    # Allow running this file directly — delegates to compare_stats.main()
    try:
        from vla_foundry.tri.utils.compare_stats import main
    except ImportError:
        from compare_stats import main
    main()
