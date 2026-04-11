#!/usr/bin/env python3
"""
Script to gather results from demonstration directories.

This script processes directories matching 'demonstration_*' pattern and extracts
success rates and trial durations from their YAML summary files.

Usage:
    uv run vla_foundry/tri/utils/gather_results.py <folder_path>

Controls:
    Arrow Keys / j/k : Navigate up/down
    Enter            : View demonstration details
    s                : Show success/failure summary
    h                : Show duration histogram
    f                : Filter by success/failure
    /                : Search demonstrations
    r                : Reset filters
    q / ESC          : Quit
    ?                : Help
"""

import argparse
import ast
import contextlib
import curses
import glob
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm


class DemoResult:
    """Container for demonstration result data."""

    def __init__(
        self,
        demo_name: str,
        success: bool,
        duration: float,
        yaml_path: str,
        video_path: str | None = None,
        yaml_data: dict[str, Any] | None = None,
        task_name: str | None = None,
        model_name: str | None = None,
    ):
        self.demo_name = demo_name
        self.success = success
        self.duration = duration
        self.yaml_path = yaml_path
        self.video_path = video_path
        self.yaml_data = yaml_data or {}
        self.task_name = task_name
        self.model_name = model_name
        self.demo_id = int(demo_name.split("_")[-1]) if "_" in demo_name else 0


class DemoResultsViewer:
    """Interactive curses-based demonstration results viewer."""

    def __init__(self, results: list[DemoResult], base_folder: str):
        self.results = results
        self.base_folder = base_folder
        self.filtered_results = results.copy()
        self.current_row = 0
        self.scroll_offset = 0
        self.search_term = ""
        self.success_filter = None  # None, True, False
        self.grouping_mode = "none"  # "none", "task", "model"
        self.view_mode = "list"  # "list", "groups"
        self.selected_group = None  # Current group we are drilled down into
        self.parent_group_filter = {}  # Filter for parent group when in subgroup view
        self.sub_grouping_mode = "none"  # Grouping mode for subgroups
        self.current_list_filters = {}  # Active filters for list view (task/model)

        # Calculate statistics
        self.success_count = sum(1 for r in results if r.success)
        self.failure_count = len(results) - self.success_count
        self.success_rate = (self.success_count / len(results) * 100) if results else 0

        durations = [r.duration for r in results]
        self.duration_stats = {
            "mean": np.mean(durations) if durations else 0,
            "median": np.median(durations) if durations else 0,
            "min": np.min(durations) if durations else 0,
            "max": np.max(durations) if durations else 0,
            "std": np.std(durations) if durations else 0,
        }

        success_durations = [r.duration for r in results if r.success]
        failure_durations = [r.duration for r in results if not r.success]

        self.success_duration_stats = {
            "mean": np.mean(success_durations) if success_durations else 0,
            "median": np.median(success_durations) if success_durations else 0,
            "count": len(success_durations),
        }

        self.failure_duration_stats = {
            "mean": np.mean(failure_durations) if failure_durations else 0,
            "median": np.median(failure_durations) if failure_durations else 0,
            "count": len(failure_durations),
        }

    def safe_addstr(self, stdscr, y: int, x: int, text: str, attr: int = 0):
        """Safely add string to screen, avoiding curses errors at edges."""
        try:
            if attr:
                stdscr.addstr(y, x, text, attr)
            else:
                stdscr.addstr(y, x, text)
        except curses.error:
            pass

    def _toggle_grouping(self):
        """Toggle between grouping modes."""
        # Reset drill-down state
        self.parent_group_filter = {}
        self.current_list_filters = {}
        self.sub_grouping_mode = "none"
        modes = ["none", "task", "model"]
        current_idx = modes.index(self.grouping_mode)
        next_mode = modes[(current_idx + 1) % len(modes)]

        self.grouping_mode = next_mode
        self.selected_group = None

        if next_mode == "none":
            self.view_mode = "list"
            self.filtered_results = self.results.copy()
            self._apply_filters()  # Re-apply search/success filters
        else:
            self.view_mode = "groups"

        # Reset scroll
        self.current_row = 0
        self.scroll_offset = 0

    def _get_sorted_results(self) -> list[DemoResult]:
        """Get results sorted according to current grouping mode."""
        if self.grouping_mode == "task":
            return sorted(self.filtered_results, key=lambda r: (r.task_name or "", r.demo_id))
        elif self.grouping_mode == "model":
            return sorted(
                self.filtered_results, key=lambda r: (r.model_name or "Unknown", r.task_name or "", r.demo_id)
            )
        return self.filtered_results

    def _get_group_stats(
        self, results: list[DemoResult] | None = None, grouping_mode: str | None = None
    ) -> list[dict[str, Any]]:
        """Calculate statistics for each group based on current grouping mode."""
        grouping_mode = grouping_mode or self.grouping_mode
        if grouping_mode == "none":
            return []

        source_results = results if results is not None else self.results

        groups = {}
        for r in source_results:
            # Fix ternary operator (SIM108) while we are here
            key = (r.task_name or "Unknown Task") if grouping_mode == "task" else (r.model_name or "Unknown Model")

            if key not in groups:
                groups[key] = {"name": key, "total": 0, "success": 0, "durations": []}

            groups[key]["total"] += 1
            if r.success:
                groups[key]["success"] += 1
            groups[key]["durations"].append(r.duration)

        # Convert to list and calculate rates
        group_list = []
        for name, data in groups.items():
            total = data["total"]
            success = data["success"]
            rate = (success / total * 100) if total > 0 else 0
            durations = data["durations"]
            mean_duration = np.mean(durations) if durations else 0

            group_list.append(
                {"name": name, "total": total, "success": success, "rate": rate, "mean_duration": mean_duration}
            )

        # Sort by name
        return sorted(group_list, key=lambda x: x["name"])

    def run(self, stdscr):
        """Main curses loop."""
        curses.curs_set(0)  # Hide cursor
        curses.use_default_colors()

        # Initialize color pairs
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)  # Selected
        curses.init_pair(2, curses.COLOR_CYAN, -1)  # Title
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # Demo name
        curses.init_pair(4, curses.COLOR_GREEN, -1)  # Success
        curses.init_pair(5, curses.COLOR_RED, -1)  # Failure
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # Duration
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_GREEN)  # Success highlight
        curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_RED)  # Failure highlight

        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()

            # Draw header
            self._draw_header(stdscr, width)

            # Draw demonstration list
            self._draw_demo_list(stdscr, height, width)

            # Draw footer
            self._draw_footer(stdscr, height, width)

            stdscr.refresh()

            # Handle input
            key = stdscr.getch()

            if key in [ord("q"), ord("Q"), 27]:  # q or ESC
                if self.view_mode == "list" and self.grouping_mode != "none":
                    if self.sub_grouping_mode != "none" and self.parent_group_filter:
                        # Go back to subgroups
                        self.view_mode = "subgroups"
                        self.current_list_filters = {}
                        self.current_row = 0
                        self.scroll_offset = 0
                        self._apply_filters()
                    else:
                        # Go back to groups view
                        self.view_mode = "groups"
                        self.selected_group = None
                        self.current_list_filters = {}
                        self.current_row = 0
                        self.scroll_offset = 0
                        self._apply_filters()
                elif self.view_mode == "subgroups":
                    self.view_mode = "groups"
                    self.parent_group_filter = {}
                    self.sub_grouping_mode = "none"
                    self.current_row = 0
                    self.scroll_offset = 0
                else:
                    break

            # Key handling need to account for view_items length
            current_view_len = len(getattr(self, "current_view_items", self.filtered_results))

            if key == ord("/"):  # Search
                self._show_search(stdscr)
            elif key in [ord("g"), ord("G")]:  # Grouping
                self._toggle_grouping()
            elif key in [ord("f"), ord("F")]:  # Filter by success/failure
                self._toggle_success_filter()
            elif key in [ord("r"), ord("R")]:  # Reset filters
                self._reset_filters()
            elif key in [ord("s"), ord("S")]:  # Show summary
                self._show_summary(stdscr)
            elif key in [ord("h"), ord("H")]:  # Show histogram
                self._show_histogram(stdscr)
            elif key in [ord("v"), ord("V")]:  # Show video
                if self.view_mode == "list" and hasattr(self, "current_view_items"):
                    result = self.current_view_items[self.current_row]
                    if result.video_path:
                        self._show_video(stdscr, result)
            elif key == ord("\n"):  # Enter
                if self.view_mode == "groups":
                    # Drill down into subgroup
                    if hasattr(self, "current_view_items") and self.current_row < len(self.current_view_items):
                        group = self.current_view_items[self.current_row]
                        group_name = group["name"]
                        self.selected_group = group_name

                        # Setup parent filter for next level
                        self.parent_group_filter = {}
                        if self.grouping_mode == "task":
                            self.parent_group_filter["task"] = group_name
                            self.sub_grouping_mode = "model"
                        else:
                            self.parent_group_filter["model"] = group_name
                            self.sub_grouping_mode = "task"

                        self.view_mode = "subgroups"
                        self.current_row = 0
                        self.scroll_offset = 0

                elif self.view_mode == "subgroups":
                    # Drill down into list
                    if hasattr(self, "current_view_items") and self.current_row < len(self.current_view_items):
                        group = self.current_view_items[self.current_row]
                        group_name = group["name"]

                        # Combine parent and current filters
                        filters = self.parent_group_filter.copy()
                        filters["task" if self.sub_grouping_mode == "task" else "model"] = group_name

                        self.current_list_filters = filters
                        self.view_mode = "list"
                        self.current_row = 0
                        self.scroll_offset = 0
                        self._apply_filters()

                else:
                    # Show details
                    if hasattr(self, "current_view_items") and self.current_row < len(self.current_view_items):
                        result = self.current_view_items[self.current_row]
                        self._show_demo_details(stdscr, result)

            elif key in [curses.KEY_DOWN, ord("j")]:
                if self.current_row < current_view_len - 1:
                    self.current_row += 1
                    if self.current_row - self.scroll_offset >= height - 6:
                        self.scroll_offset += 1
            elif key in [curses.KEY_UP, ord("k")]:
                if self.current_row > 0:
                    self.current_row -= 1
                    if self.current_row < self.scroll_offset:
                        self.scroll_offset -= 1
            elif key == curses.KEY_PPAGE:  # Page Up
                self.current_row = max(0, self.current_row - (height - 6))
                self.scroll_offset = max(0, self.scroll_offset - (height - 6))
            elif key == curses.KEY_NPAGE:  # Page Down
                self.current_row = min(current_view_len - 1, self.current_row + (height - 6))
                max_offset = max(0, current_view_len - (height - 6))
                self.scroll_offset = min(max_offset, self.scroll_offset + (height - 6))
            elif key == curses.KEY_HOME:
                self.current_row = 0
                self.scroll_offset = 0
            elif key == curses.KEY_END:
                self.current_row = current_view_len - 1
                self.scroll_offset = max(0, current_view_len - (height - 6))
            elif key in [ord("?")]:
                self._show_help(stdscr)

    def _draw_header(self, stdscr, width: int):
        """Draw the header section."""
        title = f" Demonstration Results Viewer: {Path(self.base_folder).name} "
        subtitle = f" {self.base_folder} "

        # Title
        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        self.safe_addstr(stdscr, 0, 0, title.center(width)[:width])
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

        # Subtitle
        stdscr.attron(curses.color_pair(2))
        self.safe_addstr(stdscr, 1, 0, subtitle.center(width)[:width])
        stdscr.attroff(curses.color_pair(2))

        # Summary statistics
        info = f" Total: {len(self.results)} | Showing: {len(self.filtered_results)} | "
        info += f"Success: {self.success_count} ({self.success_rate:.1f}%) | "
        info += f"Failure: {self.failure_count} "

        if self.search_term:
            info += f"| Filter: '{self.search_term}' "

        if self.success_filter is not None:
            filter_text = "Success" if self.success_filter else "Failure"
            info += f"| Mode: {filter_text} "

        if self.view_mode == "subgroups" and self.parent_group_filter:
            parent_val = list(self.parent_group_filter.values())[0]
            info += f"| {parent_val} > Group By: {self.sub_grouping_mode.title()}"
        elif self.view_mode == "list" and self.current_list_filters:
            breadcrumbs = " > ".join(self.current_list_filters.values())
            info += f"| {breadcrumbs}"
        else:
            info += f"| Group: {self.grouping_mode.title()} (Press 'g')"

        self.safe_addstr(stdscr, 2, 0, info.center(width)[:width])
        self.safe_addstr(stdscr, 3, 0, "─" * width)

    def _draw_demo_list(self, stdscr, height: int, width: int):
        """Draw the list of demonstrations or groups."""
        list_height = height - 6

        # Determine what to show based on view mode
        if self.view_mode in ["groups", "subgroups"]:
            self._draw_group_list(stdscr, list_height, width)
            return

        # Regular list view (filtered results)
        self._draw_regular_list(stdscr, list_height, width)

    def _draw_group_list(self, stdscr, list_height: int, width: int):
        """Draw list of groups."""
        if self.view_mode == "subgroups":
            results = self._filter_list(self.results, self.parent_group_filter)
            groups = self._get_group_stats(results, self.sub_grouping_mode)
            group_type = "Task" if self.sub_grouping_mode == "task" else "Model"
        else:
            groups = self._get_group_stats(self.results, self.grouping_mode)
            group_type = "Task" if self.grouping_mode == "task" else "Model"

        self.current_view_items = groups  # Store for input handling

        # Header
        header_y = 4
        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        header = f"{group_type:<40} {'Success':>10} {'Total':>8} {'Rate':>8} {'Mean Dur':>10}"
        self.safe_addstr(stdscr, header_y, 0, header[:width])
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

        # Rows
        for idx in range(list_height - 1):
            result_idx = idx + self.scroll_offset
            if result_idx >= len(groups):
                break

            group = groups[result_idx]
            y = header_y + 1 + idx

            name = group["name"][:39]
            success = group["success"]
            total = group["total"]
            rate = group["rate"]
            dur = group["mean_duration"]

            row_str = f"{name:<40} {success:>10} {total:>8} {rate:>7.1f}% {dur:>9.2f}s"

            # Highlight selected
            if result_idx == self.current_row:
                stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                self.safe_addstr(stdscr, y, 0, row_str.ljust(width)[:width])
                stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
            else:
                self.safe_addstr(stdscr, y, 0, row_str[:width])

    def _draw_regular_list(self, stdscr, list_height: int, width: int):
        """Draw list of individual demonstrations."""
        # Get sorted results based on grouping (if any, though in drilled down mode we might just sort by id)
        # If we are drilled down, we are showing self.filtered_results which should be set to the group

        # display_results = self._get_sorted_results()
        # Logic change: if view_mode is 'list', self.filtered_results IS what we want to show.
        # Sorting might still be useful.

        display_results = self.filtered_results  # Already filtered for group if applicable
        self.current_view_items = display_results

        # Column headers
        header_y = 4
        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)

        # Check if we should show Task/Model columns
        # If we are in "none" grouping mode, show them if multiple exist
        # If we are inside a group, we probably don't need to show the column for that group type

        unique_tasks = set(r.task_name for r in display_results if r.task_name)
        show_task = len(unique_tasks) > 1

        unique_models = set(r.model_name for r in display_results if r.model_name)
        show_model = len(unique_models) > 1

        header = f"{'Demo ID':>8} {'Success':>8} {'Duration':>10} {'Video':>5} "
        if show_model:
            header += f"{'Model':<20} "
        if show_task:
            header += f"{'Task':<25} "

        header += f"{'Demo Name'}"

        self.safe_addstr(stdscr, header_y, 0, header[:width])
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

        # Demonstration rows
        for idx in range(list_height - 1):
            result_idx = idx + self.scroll_offset
            if result_idx >= len(display_results):
                break

            result = display_results[result_idx]
            y = header_y + 1 + idx

            # Format row
            success_text = "✓" if result.success else "✗"
            duration_text = f"{result.duration:.2f}s"
            video_text = "📹" if result.video_path else " "

            row_str = f"{result.demo_id:>8} {success_text:>8} {duration_text:>10} {video_text:>5} "

            if show_model:
                model_str = (result.model_name or "")[:19]
                row_str += f"{model_str:<20} "
            if show_task:
                task_str = (result.task_name or "")[:24]
                row_str += f"{task_str:<25} "

            row_str += f"{result.demo_name}"

            # Highlight selected row
            if result_idx == self.current_row:
                stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                self.safe_addstr(stdscr, y, 0, row_str.ljust(width)[:width])
                stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
            else:
                self.safe_addstr(stdscr, y, 0, row_str[:width], 0)

                # Overwrite success char with color
                status_attr = curses.color_pair(4) if result.success else curses.color_pair(5)
                self.safe_addstr(stdscr, y, 9, success_text, status_attr)

    def _draw_footer(self, stdscr, height: int, width: int):
        """Draw the footer with controls."""
        if height < 4:
            return

        footer_y = height - 2
        self.safe_addstr(stdscr, footer_y, 0, "─" * width)

        controls = (
            " \u2191/\u2193:Nav | Enter:Details | v:Video | s:Summary | h:Histogram | "
            "f:Filter | g:Group | /:Search | r:Reset | ?:Help | q:Quit "
        )
        stdscr.attron(curses.A_BOLD)
        self.safe_addstr(stdscr, footer_y + 1, 0, controls.center(width)[:width])
        stdscr.attroff(curses.A_BOLD)

    def _show_search(self, stdscr):
        """Show search input dialog."""
        height, width = stdscr.getmaxyx()

        # Draw search prompt
        prompt_y = height // 2
        stdscr.clear()
        self.safe_addstr(stdscr, prompt_y - 1, 0, " " * width)
        self.safe_addstr(stdscr, prompt_y, 0, " Search Demonstrations: ".center(width))
        self.safe_addstr(stdscr, prompt_y + 1, 0, " " * width)

        # Enable echo and cursor
        curses.echo()
        curses.curs_set(1)

        # Get input
        search_str = ""
        with contextlib.suppress(Exception):
            search_str = stdscr.getstr(prompt_y, width // 2 - 10, 20).decode("utf-8")

        # Disable echo and cursor
        curses.noecho()
        curses.curs_set(0)

        # Apply filter
        self.search_term = search_str.strip()
        self._apply_filters()

    def _toggle_success_filter(self):
        """Toggle between showing all, success only, or failure only."""
        if self.success_filter is None:
            self.success_filter = True  # Show success only
        elif self.success_filter is True:
            self.success_filter = False  # Show failure only
        else:
            self.success_filter = None  # Show all

        self._apply_filters()

    def _reset_filters(self):
        """Reset all filters."""
        self.search_term = ""
        self.success_filter = None
        self._apply_filters()

    def _filter_list(self, source_list: list[DemoResult], filters: dict[str, str]) -> list[DemoResult]:
        """Filter a list of results based on a dictionary of filters."""
        filtered = source_list
        for key, value in filters.items():
            if key == "task":
                filtered = [r for r in filtered if (r.task_name or "Unknown Task") == value]
            elif key == "model":
                filtered = [r for r in filtered if (r.model_name or "Unknown Model") == value]
        return filtered

    def _apply_filters(self):
        """Apply current search and success filters."""
        if self.current_list_filters:
            self.filtered_results = self._filter_list(self.results, self.current_list_filters)
        else:
            self.filtered_results = self.results.copy()

        # Apply search filter
        if self.search_term:
            self.filtered_results = [
                r for r in self.filtered_results if self.search_term.lower() in r.demo_name.lower()
            ]

        # Apply success filter
        if self.success_filter is not None:
            self.filtered_results = [r for r in self.filtered_results if r.success == self.success_filter]

    def _get_scored_results_for_stats(self) -> list[DemoResult]:
        """Get the list of results to calculate stats/histograms for."""
        if self.view_mode == "groups" and self.grouping_mode != "none":
            # Highlighting a group in main group view
            groups = self._get_group_stats(self.results, self.grouping_mode)
            if self.current_row < len(groups):
                group_name = groups[self.current_row]["name"]
                filters = {"task" if self.grouping_mode == "task" else "model": group_name}
                return self._filter_list(self.results, filters)
            return []

        elif self.view_mode == "subgroups" and self.sub_grouping_mode != "none":
            # Highlighting a subgroup in subgroup view
            base_results = self._filter_list(self.results, self.parent_group_filter)
            groups = self._get_group_stats(base_results, self.sub_grouping_mode)
            if self.current_row < len(groups):
                group_name = groups[self.current_row]["name"]
                filters = self.parent_group_filter.copy()
                filters["task" if self.sub_grouping_mode == "task" else "model"] = group_name
                return self._filter_list(self.results, filters)
            return []

        else:
            # List mode or no grouping: use filtered_results
            return self.filtered_results

    def _show_summary(self, stdscr):
        """Show detailed detailed summary statistics for current view."""
        target_results = self._get_scored_results_for_stats()

        # Calculate stats dynamically
        total_count = len(target_results)
        success_count = sum(1 for r in target_results if r.success)
        failure_count = total_count - success_count
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0

        durations = [r.duration for r in target_results]

        duration_stats = {
            "mean": np.mean(durations) if durations else 0,
            "median": np.median(durations) if durations else 0,
            "min": np.min(durations) if durations else 0,
            "max": np.max(durations) if durations else 0,
            "std": np.std(durations) if durations else 0,
        }

        success_durations = [r.duration for r in target_results if r.success]
        success_duration_stats = {
            "mean": np.mean(success_durations) if success_durations else 0,
            "median": np.median(success_durations) if success_durations else 0,
            "count": len(success_durations),
        }

        failure_durations = [r.duration for r in target_results if not r.success]
        failure_duration_stats = {
            "mean": np.mean(failure_durations) if failure_durations else 0,
            "median": np.median(failure_durations) if failure_durations else 0,
            "count": len(failure_durations),
        }

        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()

            # Header
            header_text = " Summary Statistics "
            if self.view_mode == "groups" and self.grouping_mode != "none":
                groups = self._get_group_stats()
                if self.current_row < len(groups):
                    group_name = groups[self.current_row]["name"]
                    header_text += f"({group_name}) "

            stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
            self.safe_addstr(stdscr, 0, 0, header_text.center(width)[:width])
            stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            self.safe_addstr(stdscr, 1, 0, "─" * width)

            # Statistics content
            lines = []
            lines.append(("", 0))
            lines.append(("OVERALL RESULTS:", 2))
            lines.append((f"  Total demonstrations: {total_count}", 0))
            lines.append((f"  Successful: {success_count}", 4))
            lines.append((f"  Failed: {failure_count}", 5))
            lines.append((f"  Success rate: {success_rate:.2f}%", 4 if success_rate > 50 else 5))
            lines.append(("", 0))

            lines.append(("DURATION STATISTICS:", 2))
            lines.append((f"  Mean duration: {duration_stats['mean']:.2f}s", 0))
            lines.append((f"  Median duration: {duration_stats['median']:.2f}s", 0))
            lines.append((f"  Min duration: {duration_stats['min']:.2f}s", 0))
            lines.append((f"  Max duration: {duration_stats['max']:.2f}s", 0))
            lines.append((f"  Std deviation: {duration_stats['std']:.2f}s", 0))
            lines.append(("", 0))

            if success_duration_stats["count"] > 0:
                lines.append(("SUCCESSFUL TRIALS:", 4))
                lines.append((f"  Count: {success_duration_stats['count']}", 0))
                lines.append((f"  Mean duration: {success_duration_stats['mean']:.2f}s", 0))
                lines.append((f"  Median duration: {success_duration_stats['median']:.2f}s", 0))
                lines.append(("", 0))

            if failure_duration_stats["count"] > 0:
                lines.append(("FAILED TRIALS:", 5))
                lines.append((f"  Count: {failure_duration_stats['count']}", 0))
                lines.append((f"  Mean duration: {failure_duration_stats['mean']:.2f}s", 0))
                lines.append((f"  Median duration: {failure_duration_stats['median']:.2f}s", 0))
                lines.append(("", 0))

            # Display lines
            display_height = height - 4
            for idx in range(min(display_height, len(lines))):
                line_text, color_pair = lines[idx]
                y = idx + 2

                if color_pair > 0:
                    stdscr.attron(curses.color_pair(color_pair))
                self.safe_addstr(stdscr, y, 2, line_text[: width - 4])
                if color_pair > 0:
                    stdscr.attroff(curses.color_pair(color_pair))

            # Footer
            footer_y = height - 2
            self.safe_addstr(stdscr, footer_y, 0, "─" * width)
            controls = " ESC/q:Back "
            stdscr.attron(curses.A_BOLD)
            self.safe_addstr(stdscr, footer_y + 1, 0, controls.center(width)[:width])
            stdscr.attroff(curses.A_BOLD)

            stdscr.refresh()

            # Handle input
            key = stdscr.getch()
            if key in [ord("q"), ord("Q"), 27]:
                break

    def _show_histogram(self, stdscr):
        """Show duration histogram for current view/selection."""
        target_results = self._get_scored_results_for_stats()

        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()

            # Header
            header_text = " Duration Distribution "
            if self.view_mode == "groups" and self.grouping_mode != "none":
                groups = self._get_group_stats()
                if self.current_row < len(groups):
                    group_name = groups[self.current_row]["name"]
                    header_text += f"({group_name}) "

            stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
            self.safe_addstr(stdscr, 0, 0, header_text.center(width)[:width])
            stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            self.safe_addstr(stdscr, 1, 0, "─" * width)

            # Create histogram
            all_durations = [r.duration for r in target_results]
            success_durations = [r.duration for r in target_results if r.success]
            failure_durations = [r.duration for r in target_results if not r.success]

            if not all_durations:
                msg = "No data to display"
                self.safe_addstr(stdscr, height // 2, (width - len(msg)) // 2, msg)
            else:
                hist_lines = self._create_text_histogram(all_durations, success_durations, failure_durations, width - 4)

                # Display histogram
                display_height = height - 4
                for idx in range(min(display_height, len(hist_lines))):
                    y = idx + 2
                    line_segments = hist_lines[idx]

                    current_x = 2
                    for text, color_pair in line_segments:
                        if width > current_x + len(text):
                            if color_pair > 0:
                                stdscr.attron(curses.color_pair(color_pair))

                            self.safe_addstr(stdscr, y, current_x, text)

                            if color_pair > 0:
                                stdscr.attroff(curses.color_pair(color_pair))

                            current_x += len(text)

            # Footer
            footer_y = height - 2
            self.safe_addstr(stdscr, footer_y, 0, "─" * width)
            controls = " ESC/q:Back "
            stdscr.attron(curses.A_BOLD)
            self.safe_addstr(stdscr, footer_y + 1, 0, controls.center(width)[:width])
            stdscr.attroff(curses.A_BOLD)

            stdscr.refresh()

            # Handle input
            key = stdscr.getch()
            if key in [ord("q"), ord("Q"), 27]:
                break

    def _create_text_histogram(
        self, all_durations: list[float], success_durations: list[float], failure_durations: list[float], width: int
    ) -> list[list[tuple[str, int]]]:
        """Create a text-based histogram using unicode blocks and colors."""
        lines = []

        # Helper to create a simple text line (no color/attributes)
        def simple_line(text):
            return [(text, 0)]

        if not all_durations:
            return [simple_line("No data available")]

        # Statistics header
        lines.append(simple_line(f"Total samples: {len(all_durations)}"))
        lines.append(simple_line(f"Success: {len(success_durations)}, Failure: {len(failure_durations)}"))
        lines.append(simple_line(f"Range: [{min(all_durations):.2f}s, {max(all_durations):.2f}s]"))
        lines.append(simple_line(""))

        # Create bins
        min_val = min(all_durations)
        max_val = max(all_durations)

        if min_val == max_val:
            lines.append(simple_line(f"All durations are {min_val:.2f}s"))
            return lines

        # Reserve space for labels
        chart_width = width
        num_bins = min(chart_width, 60)  # Allow more bins if space permits
        bin_width = (max_val - min_val) / num_bins

        # Count values in bins
        all_bins = [0] * num_bins
        success_bins = [0] * num_bins

        for duration in all_durations:
            bin_idx = int((duration - min_val) / bin_width)
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            all_bins[bin_idx] += 1

        for duration in success_durations:
            bin_idx = int((duration - min_val) / bin_width)
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            success_bins[bin_idx] += 1

        max_count = max(all_bins) if all_bins else 1

        # Check for outliers (specifically timeout spikes) to avoid crushing the rest of the distribution
        # Simple heuristic: if max bin is > 2.0x second max, clip it
        # Actually, let's look at all unique non-zero counts
        unique_counts = sorted(list(set([c for c in all_bins if c > 0])))
        scale_max = max_count
        clipped_bins = []

        if len(unique_counts) > 1:
            second_max = unique_counts[-2]
            if max_count > 2.0 * second_max:
                scale_max = int(second_max * 1.25)  # Give a little headroom
                # Identify clipped bins
                for idx, count in enumerate(all_bins):
                    if count > scale_max:
                        clipped_bins.append((idx, count))

        # If we are clipping, use scale_max for normalization
        display_max = scale_max

        hist_height = 10

        # Unicode blocks: empty, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, full
        # block_chars = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        block_chars = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

        lines.append(simple_line("Distribution (Green=Success, Red=Failure, Yellow=Mixed):"))

        for i in range(hist_height - 1, -1, -1):
            # Row spans from i/height to (i+1)/height in vertical space
            row_segments = []

            # Y-axis label
            label_val = int(display_max * (i + 1) / hist_height)
            # Only label top and bottom to save space? Or every line?
            # Let's label every other line
            if i % 2 == 0 or i == hist_height - 1:
                row_segments.append((f"{label_val:>4} |", 0))
            else:
                row_segments.append(("     |", 0))

            for j in range(num_bins):
                count = all_bins[j]

                # Check if this bin is clipped
                is_clipped = count > display_max
                effective_count = min(count, display_max)

                normalized_height = (effective_count / display_max) * hist_height

                # Determine how much of this specific row 'i' is filled by the bar
                # Row i corresponds to interval [i, i+1] in height units

                fill_amt = normalized_height - i

                # Determine character
                char_idx = 0
                if fill_amt <= 0:
                    char_idx = 0  # Empty
                elif fill_amt >= 1:
                    char_idx = 8  # Full block
                else:
                    char_idx = int(fill_amt * 8)
                    if char_idx < 0:
                        char_idx = 0
                    if char_idx > 8:
                        char_idx = 8
                    # If it's effectively 0 but count > 0, make sure to show at least 1/8th in bottom row
                    if i == 0 and count > 0 and char_idx == 0:
                        char_idx = 1

                char_to_draw = block_chars[char_idx]

                # Special marker for top of clipped bar
                # If we are at the top row (hist_height - 1) and this bin is clipped, use special char
                if is_clipped and i == hist_height - 1:
                    char_to_draw = "↑"  # Arrow indicating continuation
                    # If this is the last bin, we can safely append the count without breaking alignment
                    if j == num_bins - 1:
                        char_to_draw += f" {count}"

                # Determine color
                color_pair = 0
                if count > 0:
                    ratio = success_bins[j] / count
                    if ratio > 0.8:
                        color_pair = 4  # Green (Success)
                    elif ratio < 0.2:
                        color_pair = 5  # Red (Failure)
                    else:
                        color_pair = 3  # Yellow (Mixed) - usually pair 3 is yellow text

                row_segments.append((char_to_draw, color_pair))

            lines.append(row_segments)

        # X-axis
        lines.append(simple_line("     " + "─" * num_bins))

        # X-axis labels (Start and End)
        x_labels_str = f"     {min_val:.1f}"
        spacing = num_bins - len(f"{min_val:.1f}") - len(f"{max_val:.1f}")
        if spacing > 0:
            x_labels_str += " " * spacing + f"{max_val:.1f}"
        lines.append(simple_line(x_labels_str))

        # Add Clipped Values footnote if necessary
        if clipped_bins:
            lines.append(simple_line(""))
            lines.append(simple_line("Clipped values (exceeding scale):"))
            for idx, count in clipped_bins:
                bin_mid = min_val + (idx + 0.5) * bin_width
                lines.append(simple_line(f"  Bin at ~{bin_mid:.1f}s: {count}"))

        return lines

    def _show_video(self, stdscr, result: DemoResult):
        """Open video with the system's default video player."""
        if not result.video_path or not os.path.exists(result.video_path):
            # Show error message
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            msg = f"Video not found: {result.video_path or 'None'}"
            self.safe_addstr(stdscr, height // 2, (width - len(msg)) // 2, msg)
            self.safe_addstr(stdscr, height // 2 + 2, (width - 20) // 2, "Press any key...")
            stdscr.refresh()
            stdscr.getch()
            return

        try:
            # Show launching message
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            msg = "Opening video with system default player..."
            self.safe_addstr(stdscr, height // 2, (width - len(msg)) // 2, msg)
            msg2 = "Press any key after closing video player"
            self.safe_addstr(stdscr, height // 2 + 2, (width - len(msg2)) // 2, msg2)
            stdscr.refresh()

            # Reset terminal to normal mode
            curses.endwin()

            # Open with system default player
            if os.name == "posix":  # Unix-like systems (Linux, macOS)
                subprocess.run(["xdg-open", result.video_path])
            elif os.name == "nt":  # Windows
                subprocess.run(["start", result.video_path], shell=True)
            else:
                # Fallback
                subprocess.run(["open", result.video_path])  # macOS fallback

            # Restore curses mode
            stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            stdscr.keypad(True)

            # Wait for user
            stdscr.clear()
            msg = "Video opened. Press any key to continue..."
            self.safe_addstr(stdscr, height // 2, (width - len(msg)) // 2, msg)
            stdscr.refresh()
            stdscr.getch()

        except Exception as e:
            # Restore curses mode if needed
            try:
                curses.initscr()
                try:
                    curses.noecho()
                    curses.cbreak()
                    stdscr.keypad(True)
                except Exception:
                    pass
            except Exception:
                pass

            stdscr.clear()
            height, width = stdscr.getmaxyx()
            msg = f"Error opening video: {str(e)}"
            self.safe_addstr(stdscr, height // 2, (width - len(msg)) // 2, msg)
            msg2 = "Press any key to continue..."
            self.safe_addstr(stdscr, height // 2 + 2, (width - len(msg2)) // 2, msg2)
            stdscr.refresh()
            stdscr.getch()

    def _show_demo_details(self, stdscr, result: DemoResult):
        """Show detailed information about a demonstration with navigation."""
        current_index = self.current_row
        scroll = 0

        while True:
            # Get current result (may have changed due to navigation)
            if current_index < len(self.filtered_results):
                current_result = self.filtered_results[current_index]
            else:
                break  # Index out of bounds

            video_path = current_result.video_path

            # Build detailed info lines
            info_lines = []
            info_lines.append((f"Demo Details: {current_result.demo_name}", 2, True))  # (text, color, bold)
            info_lines.append(("", 0, False))

            # Basic info
            info_lines.append(("Basic Information:", 2, True))
            info_lines.append((f"  Name: {current_result.demo_name}", 0, False))
            info_lines.append((f"  Duration: {current_result.duration:.2f}s", 0, False))

            if current_result.success is not None:
                status = "Success" if current_result.success else "Failure"
                color = 4 if current_result.success else 5
                info_lines.append((f"  Status: {status}", color, False))

            info_lines.append(("", 0, False))

            # YAML data details
            if current_result.yaml_data:
                info_lines.append(("YAML Data:", 2, True))
                yaml_data = current_result.yaml_data

                # Show key fields from YAML
                for key, value in yaml_data.items():
                    if key in ["success", "last_t"]:  # Skip already shown basic info
                        continue

                    # Format value for display
                    if isinstance(value, (dict, list)):
                        if key == "env_metadata" and isinstance(value, dict):
                            info_lines.append((f"  {key}:", 3, False))
                            for subkey, subvalue in value.items():
                                info_lines.append((f"    {subkey}: {str(subvalue)[:60]}", 0, False))
                        else:
                            info_lines.append(
                                (f"  {key}: {str(value)[:60]}{'...' if len(str(value)) > 60 else ''}", 3, False)
                            )
                    else:
                        info_lines.append((f"  {key}: {str(value)}", 0, False))

                info_lines.append(("", 0, False))

            # Video info
            info_lines.append(("Video:", 2, True))
            if video_path:
                video_name = os.path.basename(video_path)
                info_lines.append((f"  File: {video_name}", 0, False))
                info_lines.append(("  Controls: v - Open with default player", 6, False))
            else:
                info_lines.append(("  No video available", 6, False))

            info_lines.append(("", 0, False))
            info_lines.append(("Navigation:", 2, True))
            info_lines.append(("  ←/→ - Previous/Next demo", 6, False))
            info_lines.append(("  ↑/↓ - Scroll content", 6, False))

            # Display content
            stdscr.clear()
            height, width = stdscr.getmaxyx()

            # Calculate scrollable area
            display_height = height - 4
            max_scroll = max(0, len(info_lines) - display_height)
            scroll = max(0, min(scroll, max_scroll))

            # Display lines with scrolling
            for idx in range(display_height):
                line_idx = idx + scroll
                if line_idx >= len(info_lines):
                    break

                text, color_pair, bold = info_lines[line_idx]
                y = idx + 2

                attrs = curses.color_pair(color_pair) if color_pair > 0 else 0
                if bold:
                    attrs |= curses.A_BOLD

                self.safe_addstr(stdscr, y, 2, text[: width - 4], attrs)

            # Header
            title = f" Demo {current_index + 1}/{len(self.filtered_results)} "
            stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
            self.safe_addstr(stdscr, 0, 0, title.center(width)[:width])
            stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            self.safe_addstr(stdscr, 1, 0, "─" * width)

            # Footer
            footer_y = height - 2
            self.safe_addstr(stdscr, footer_y, 0, "─" * width)
            controls = " ←/→:Navigate | ↑/↓:Scroll | v:Video | ESC/q:Back "
            stdscr.attron(curses.A_BOLD)
            self.safe_addstr(stdscr, footer_y + 1, 0, controls.center(width)[:width])
            stdscr.attroff(curses.A_BOLD)

            stdscr.refresh()

            # Handle input
            key = stdscr.getch()

            if key in [ord("q"), ord("Q"), 27]:  # q or ESC
                break
            elif key == curses.KEY_LEFT and current_index > 0:
                current_index -= 1
                scroll = 0  # Reset scroll when navigating
            elif key == curses.KEY_RIGHT and current_index < len(self.filtered_results) - 1:
                current_index += 1
                scroll = 0  # Reset scroll when navigating
            elif key in [curses.KEY_UP, ord("k")] and scroll > 0:
                scroll -= 1
            elif key in [curses.KEY_DOWN, ord("j")] and scroll < max_scroll:
                scroll += 1
            elif key == ord("v") and video_path:
                # Open video with system default player
                try:
                    curses.endwin()
                    subprocess.run(["xdg-open", video_path], check=True)
                    # Restore curses
                    stdscr = curses.initscr()
                    curses.noecho()
                    curses.cbreak()
                    stdscr.keypad(True)
                    if curses.has_colors():
                        curses.start_color()
                        self._init_colors()
                except Exception:
                    # Restore curses mode if needed
                    try:
                        stdscr = curses.initscr()
                        curses.noecho()
                        curses.cbreak()
                        stdscr.keypad(True)
                        try:
                            curses.start_color()
                            self._init_colors()
                        except Exception:
                            pass
                    except Exception:
                        pass

        # Update the main view's current row to reflect navigation
        self.current_row = current_index

    def _run_colored_video_player(
        self, stdscr, video_path: str, duration: float, fps: float, total_frames: int, demo_name: str
    ):
        """Run the colored video player interface."""
        current_frame = 0
        playing = False
        frame_cache = {}
        playback_speed = 1.0

        # Get original video dimensions for aspect ratio
        video_width, video_height = self._get_video_dimensions(video_path)

        # Initialize additional color pairs for colored ASCII
        try:
            # Extended color palette
            curses.init_pair(10, curses.COLOR_BLACK, -1)  # Dark
            curses.init_pair(11, curses.COLOR_BLUE, -1)  # Blue tones
            curses.init_pair(12, curses.COLOR_CYAN, -1)  # Cyan tones
            curses.init_pair(13, curses.COLOR_GREEN, -1)  # Green tones
            curses.init_pair(14, curses.COLOR_YELLOW, -1)  # Yellow tones
            curses.init_pair(15, curses.COLOR_RED, -1)
            curses.init_pair(16, curses.COLOR_MAGENTA, -1)  # Magenta tones
            curses.init_pair(17, curses.COLOR_WHITE, -1)  # White/bright
        except Exception:
            pass  # Fallback if colors not supported

        # Terminal dimensions for ASCII conversion with proper aspect ratio
        height, width = stdscr.getmaxyx()
        available_height = height - 8  # Reserve space for controls
        available_width = width - 4

        # Calculate ASCII dimensions maintaining aspect ratio
        if video_width > 0 and video_height > 0:
            aspect_ratio = video_width / video_height

            # Terminal characters are roughly 2:1 height to width ratio
            # Adjust for terminal character aspect ratio
            terminal_aspect_correction = 2.0
            adjusted_aspect = aspect_ratio / terminal_aspect_correction

            # Calculate best fit dimensions
            if adjusted_aspect > (available_width / available_height):
                # Width constrained
                ascii_width = available_width
                ascii_height = int(available_width / adjusted_aspect)
            else:
                # Height constrained
                ascii_height = available_height
                ascii_width = int(available_height * adjusted_aspect)

            # Ensure minimum size
            ascii_width = max(20, min(ascii_width, available_width))
            ascii_height = max(10, min(ascii_height, available_height))
        else:
            # Fallback if we can't get dimensions
            ascii_width = available_width
            ascii_height = available_height

        while True:
            stdscr.clear()

            # Header
            title = f" Colored Video Player: {demo_name} "
            stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
            self.safe_addstr(stdscr, 0, 0, title.center(width)[:width])
            stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

            # Video info
            current_time = current_frame / fps
            info = (
                f" Frame: {current_frame}/{total_frames} | "
                f"Time: {current_time:.2f}/{duration:.2f}s | Speed: {playback_speed:.1f}x "
            )
            if playing:
                info += "| ▶ PLAYING "
            else:
                info += "| ⏸ PAUSED "

            self.safe_addstr(stdscr, 1, 0, info.center(width)[:width])
            self.safe_addstr(stdscr, 2, 0, "─" * width)

            # Get and display frame
            if current_frame not in frame_cache:
                colored_ascii_frame = self._extract_colored_ascii_frame(
                    video_path, current_frame, ascii_width, ascii_height
                )
                frame_cache[current_frame] = colored_ascii_frame

            colored_ascii_frame = frame_cache[current_frame]

            # Display colored frame (centered)
            start_y = 3 + (available_height - ascii_height) // 2
            start_x = 2 + (available_width - ascii_width) // 2

            for i, (line, colors) in enumerate(colored_ascii_frame):
                if i < ascii_height and start_y + i < height - 4:
                    x_pos = start_x
                    for j, (char, color_pair) in enumerate(zip(line, colors, strict=False)):
                        if j < ascii_width and x_pos < width - 2:
                            try:
                                if color_pair > 0:
                                    stdscr.attron(curses.color_pair(color_pair))
                                self.safe_addstr(stdscr, start_y + i, x_pos, char)
                                if color_pair > 0:
                                    stdscr.attroff(curses.color_pair(color_pair))
                                x_pos += 1
                            except Exception:
                                x_pos += 1

            # Progress bar
            progress_y = height - 4
            progress_width = width - 10
            progress = current_frame / total_frames if total_frames > 0 else 0
            filled = int(progress * progress_width)

            progress_bar = f"[{'█' * filled}{'─' * (progress_width - filled)}]"
            self.safe_addstr(stdscr, progress_y, 2, progress_bar)

            # Controls
            controls_y = height - 2
            self.safe_addstr(stdscr, controls_y, 0, "─" * width)
            controls = " Space:Play/Pause | ←/→:Frame | ↑/↓:Speed | Home/End:Jump | q:Exit "
            stdscr.attron(curses.A_BOLD)
            self.safe_addstr(stdscr, controls_y + 1, 0, controls.center(width)[:width])
            stdscr.attroff(curses.A_BOLD)

            stdscr.refresh()

            # Handle input (same as original)
            if playing:
                stdscr.nodelay(True)
                key = stdscr.getch()
                stdscr.nodelay(False)

                if key == -1:  # No key pressed, advance frame
                    time.sleep(1.0 / (fps * playback_speed))
                    current_frame = min(current_frame + 1, total_frames - 1)
                    if current_frame >= total_frames - 1:
                        playing = False
                else:
                    # Handle key while playing
                    if key == ord(" "):
                        playing = False
                    elif key in [ord("q"), ord("Q"), 27]:
                        break
                    elif key == curses.KEY_LEFT:
                        current_frame = max(0, current_frame - 1)
                    elif key == curses.KEY_RIGHT:
                        current_frame = min(total_frames - 1, current_frame + 1)
                    elif key == curses.KEY_UP:
                        playback_speed = min(4.0, playback_speed * 1.5)
                    elif key == curses.KEY_DOWN:
                        playback_speed = max(0.25, playback_speed / 1.5)
                    elif key == curses.KEY_HOME:
                        current_frame = 0
                    elif key == curses.KEY_END:
                        current_frame = total_frames - 1
                        playing = False
            else:
                # Paused - wait for input
                key = stdscr.getch()

                if key == ord(" "):
                    playing = True
                elif key in [ord("q"), ord("Q"), 27]:
                    break
                elif key == curses.KEY_LEFT:
                    current_frame = max(0, current_frame - 1)
                elif key == curses.KEY_RIGHT:
                    current_frame = min(total_frames - 1, current_frame + 1)
                elif key == curses.KEY_UP:
                    playback_speed = min(4.0, playback_speed * 1.5)
                elif key == curses.KEY_DOWN:
                    playback_speed = max(0.25, playback_speed / 1.5)
                elif key == curses.KEY_HOME:
                    current_frame = 0
                elif key == curses.KEY_END:
                    current_frame = total_frames - 1
                elif key in [ord("j")]:
                    current_frame = max(0, current_frame - int(fps))  # Jump back 1 second
                elif key in [ord("l")]:
                    current_frame = min(total_frames - 1, current_frame + int(fps))  # Jump forward 1 second

    def _get_video_dimensions(self, video_path: str) -> tuple[int, int]:
        """Get the original width and height of the video."""
        try:
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json

            video_info = json.loads(result.stdout)

            # Find video stream
            for stream in video_info["streams"]:
                if stream["codec_type"] == "video":
                    width = int(stream.get("width", 0))
                    height = int(stream.get("height", 0))
                    return width, height

            return 0, 0  # No video stream found

        except Exception:
            return 0, 0  # Fallback

    def _extract_colored_ascii_frame(
        self, video_path: str, frame_number: int, width: int, height: int
    ) -> list[tuple[str, list[int]]]:
        """Extract a single frame and convert to colored ASCII art."""
        try:
            # Create temporary file for frame
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            # Extract frame using ffmpeg
            cmd = ["ffmpeg", "-i", video_path, "-vf", f"select=eq(n\\,{frame_number})", "-vframes", "1", "-y", tmp_path]

            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if result.returncode != 0 or not os.path.exists(tmp_path):
                return [(f"Error extracting frame {frame_number}", [0] * len(f"Error extracting frame {frame_number}"))]

            # Convert to colored ASCII
            colored_ascii_frame = self._image_to_colored_ascii(tmp_path, width, height)

            # Clean up
            os.unlink(tmp_path)

            return colored_ascii_frame

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return [(error_msg, [0] * len(error_msg))]

    def _image_to_colored_ascii(self, image_path: str, width: int, height: int) -> list[tuple[str, list[int]]]:
        """Convert image to colored ASCII art."""
        try:
            # Try to use PIL if available
            try:
                from PIL import Image

                # Open and resize image
                img = Image.open(image_path)

                # Keep color information
                img_rgb = img.convert("RGB")
                img_gray = img.convert("L")  # Also get grayscale for character selection

                img_rgb = img_rgb.resize((width, height), Image.LANCZOS)
                img_gray = img_gray.resize((width, height), Image.LANCZOS)

                # ASCII characters from dark to light with more variety for detail
                ascii_chars = " ░▒▓█"
                # Alternative with more detail: " .·:;=*#%@"

                colored_lines = []
                for y in range(height):
                    line = ""
                    colors = []
                    for x in range(width):
                        # Get color and brightness
                        r, g, b = img_rgb.getpixel((x, y))
                        gray = img_gray.getpixel((x, y))

                        # Select character based on brightness
                        char_index = int((gray / 255) * (len(ascii_chars) - 1))
                        char = ascii_chars[char_index]

                        # Determine color based on RGB values
                        color_pair = self._rgb_to_color_pair(r, g, b, gray)

                        line += char
                        colors.append(color_pair)

                    colored_lines.append((line, colors))

                return colored_lines

            except ImportError:
                # Fallback to basic ASCII without color
                return self._basic_ascii_fallback(image_path, width, height)

        except Exception as e:
            error_msg = f"Error converting image: {str(e)}"
            return [(error_msg, [0] * len(error_msg))]

    def _rgb_to_color_pair(self, r: int, g: int, b: int, brightness: int) -> int:
        """Convert RGB values to appropriate color pair index."""
        try:
            # Normalize RGB values
            total = r + g + b
            if total == 0:
                return 10  # Black

            r_norm = r / total
            g_norm = g / total
            b_norm = b / total

            # Determine dominant color with brightness consideration
            if brightness < 50:  # Very dark
                return 10  # Black
            elif brightness > 200:  # Very bright
                return 17  # White
            elif r_norm > 0.5 and r_norm > g_norm and r_norm > b_norm:
                return 15  # Red
            elif g_norm > 0.4 and g_norm > b_norm:
                return 13  # Green
            elif b_norm > 0.4:
                return 11  # Blue
            elif r_norm > 0.35 and g_norm > 0.35:
                return 14  # Yellow
            elif r_norm > 0.3 and b_norm > 0.3:
                return 16  # Magenta
            elif g_norm > 0.3 and b_norm > 0.3:
                return 12  # Cyan
            else:
                # Use brightness to select grayscale
                if brightness > 128:
                    return 17  # White
                else:
                    return 10  # Black
        except Exception:
            return 0  # Default

    def _basic_ascii_fallback(self, image_path: str, width: int, height: int) -> list[tuple[str, list[int]]]:
        """Fallback ASCII conversion without color."""
        try:
            # Use ImageMagick if available
            cmd = ["convert", image_path, "-resize", f"{width}x{height}!", "-colorspace", "Gray", "txt:-"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                ascii_chars = " .:-=+*#%@"
                colored_lines = []

                for i in range(0, min(len(lines), height * width), width):
                    line = ""
                    colors = []
                    for j in range(width):
                        if i + j < len(lines):
                            # Extract brightness from ImageMagick output
                            parts = lines[i + j].split()
                            if len(parts) >= 3:
                                gray_val = int(parts[2].split(",")[0].replace("(", ""))
                                char_index = int((gray_val / 255) * (len(ascii_chars) - 1))
                                char = ascii_chars[char_index]
                                # Simple color based on brightness
                                if gray_val > 200:
                                    color = 17
                                elif gray_val > 150:
                                    color = 14
                                elif gray_val > 100:
                                    color = 13
                                elif gray_val > 50:
                                    color = 11
                                else:
                                    color = 10
                            else:
                                char = " "
                                color = 0
                        else:
                            char = " "
                            color = 0
                        line += char
                        colors.append(color)
                    colored_lines.append((line, colors))
                    if len(colored_lines) >= height:
                        break

                return colored_lines[:height]

        except FileNotFoundError:
            pass

        # Ultimate fallback - basic monochrome
        error_msg = "Color conversion requires PIL (pip install Pillow) or ImageMagick"
        return [(error_msg, [0] * len(error_msg))]

    def _show_ascii_video(self, stdscr, video_path: str, demo_name: str):
        """Show ASCII art video (original implementation)."""
        # Try to get video info
        try:
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path]
            result_probe = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json

            video_info = json.loads(result_probe.stdout)

            # Extract frame rate and duration
            duration = float(video_info["format"]["duration"])

            # Find video stream
            video_stream = None
            for stream in video_info["streams"]:
                if stream["codec_type"] == "video":
                    video_stream = stream
                    break

            if not video_stream:
                raise Exception("No video stream found")

            fps = eval(video_stream.get("r_frame_rate", "30/1"))  # Convert fraction to float
            total_frames = int(float(video_stream.get("nb_frames", duration * fps)))

        except Exception:
            # Fallback values - try to estimate from file
            try:
                # Get basic duration
                cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", video_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                duration = float(result.stdout.strip()) if result.stdout.strip() else 60.0
            except Exception:
                duration = 60.0  # Default fallback

            fps = 30.0
            total_frames = int(duration * fps)

        # Video player interface with proper aspect ratio
        self._run_video_player(stdscr, video_path, duration, fps, total_frames, demo_name)

    def _run_video_player(
        self, stdscr, video_path: str, duration: float, fps: float, total_frames: int, demo_name: str
    ):
        """Run the video player interface."""
        current_frame = 0
        playing = False
        frame_cache = {}
        playback_speed = 1.0

        # Get original video dimensions for aspect ratio
        video_width, video_height = self._get_video_dimensions(video_path)

        # Terminal dimensions for ASCII conversion with proper aspect ratio
        height, width = stdscr.getmaxyx()
        available_height = height - 8  # Reserve space for controls
        available_width = width - 4

        # Calculate ASCII dimensions maintaining aspect ratio
        if video_width > 0 and video_height > 0:
            aspect_ratio = video_width / video_height

            # Terminal characters are roughly 2:1 height to width ratio
            terminal_aspect_correction = 2.0
            adjusted_aspect = aspect_ratio / terminal_aspect_correction

            # Calculate best fit dimensions
            if adjusted_aspect > (available_width / available_height):
                # Width constrained
                ascii_width = available_width
                ascii_height = int(available_width / adjusted_aspect)
            else:
                # Height constrained
                ascii_height = available_height
                ascii_width = int(available_height * adjusted_aspect)

            # Ensure minimum size
            ascii_width = max(20, min(ascii_width, available_width))
            ascii_height = max(10, min(ascii_height, available_height))
        else:
            # Fallback if we can't get dimensions
            ascii_width = available_width
            ascii_height = available_height

        while True:
            stdscr.clear()

            # Header
            title = f" Video Player: {demo_name} "
            stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
            self.safe_addstr(stdscr, 0, 0, title.center(width)[:width])
            stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

            # Video info
            current_time = current_frame / fps
            info = (
                f" Frame: {current_frame}/{total_frames} | "
                f"Time: {current_time:.2f}/{duration:.2f}s | Speed: {playback_speed:.1f}x "
            )
            if playing:
                info += "| ▶ PLAYING "
            else:
                info += "| ⏸ PAUSED "

            self.safe_addstr(stdscr, 1, 0, info.center(width)[:width])
            self.safe_addstr(stdscr, 2, 0, "─" * width)

            # Get and display frame
            if current_frame not in frame_cache:
                ascii_frame = self._extract_ascii_frame(video_path, current_frame, ascii_width, ascii_height)
                frame_cache[current_frame] = ascii_frame

            ascii_frame = frame_cache[current_frame]

            # Display frame (centered)
            start_y = 3 + (available_height - ascii_height) // 2
            start_x = 2 + (available_width - ascii_width) // 2

            for i, line in enumerate(ascii_frame):
                if i < ascii_height and start_y + i < height - 4:
                    display_line = line[:ascii_width] if len(line) > ascii_width else line
                    self.safe_addstr(stdscr, start_y + i, start_x, display_line)

            # Progress bar
            progress_y = height - 4
            progress_width = width - 10
            progress = current_frame / total_frames if total_frames > 0 else 0
            filled = int(progress * progress_width)

            progress_bar = f"[{'█' * filled}{'─' * (progress_width - filled)}]"
            self.safe_addstr(stdscr, progress_y, 2, progress_bar)

            # Controls
            controls_y = height - 2
            self.safe_addstr(stdscr, controls_y, 0, "─" * width)
            controls = " Space:Play/Pause | ←/→:Frame | ↑/↓:Speed | Home/End:Jump | q:Exit "
            stdscr.attron(curses.A_BOLD)
            self.safe_addstr(stdscr, controls_y + 1, 0, controls.center(width)[:width])
            stdscr.attroff(curses.A_BOLD)

            stdscr.refresh()

            # Handle input
            if playing:
                stdscr.nodelay(True)
                key = stdscr.getch()
                stdscr.nodelay(False)

                if key == -1:  # No key pressed, advance frame
                    time.sleep(1.0 / (fps * playback_speed))
                    current_frame = min(current_frame + 1, total_frames - 1)
                    if current_frame >= total_frames - 1:
                        playing = False
                else:
                    # Handle key while playing
                    if key == ord(" "):
                        playing = False
                    elif key in [ord("q"), ord("Q"), 27]:
                        break
                    elif key == curses.KEY_LEFT:
                        current_frame = max(0, current_frame - 1)
                    elif key == curses.KEY_RIGHT:
                        current_frame = min(total_frames - 1, current_frame + 1)
                    elif key == curses.KEY_UP:
                        playback_speed = min(4.0, playback_speed * 1.5)
                    elif key == curses.KEY_DOWN:
                        playback_speed = max(0.25, playback_speed / 1.5)
                    elif key == curses.KEY_HOME:
                        current_frame = 0
                    elif key == curses.KEY_END:
                        current_frame = total_frames - 1
                        playing = False
            else:
                # Paused - wait for input
                key = stdscr.getch()

                if key == ord(" "):
                    playing = True
                elif key in [ord("q"), ord("Q"), 27]:
                    break
                elif key == curses.KEY_LEFT:
                    current_frame = max(0, current_frame - 1)
                elif key == curses.KEY_RIGHT:
                    current_frame = min(total_frames - 1, current_frame + 1)
                elif key == curses.KEY_UP:
                    playback_speed = min(4.0, playback_speed * 1.5)
                elif key == curses.KEY_DOWN:
                    playback_speed = max(0.25, playback_speed / 1.5)
                elif key == curses.KEY_HOME:
                    current_frame = 0
                elif key == curses.KEY_END:
                    current_frame = total_frames - 1
                elif key in [ord("j")]:
                    current_frame = max(0, current_frame - int(fps))  # Jump back 1 second
                elif key in [ord("l")]:
                    current_frame = min(total_frames - 1, current_frame + int(fps))  # Jump forward 1 second

    def _extract_ascii_frame(self, video_path: str, frame_number: int, width: int, height: int) -> list[str]:
        """Extract a single frame and convert to ASCII art."""
        try:
            # Create temporary file for frame
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            # Extract frame using ffmpeg
            cmd = ["ffmpeg", "-i", video_path, "-vf", f"select=eq(n\\,{frame_number})", "-vframes", "1", "-y", tmp_path]

            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if result.returncode != 0 or not os.path.exists(tmp_path):
                return [f"Error extracting frame {frame_number}"]

            # Convert to ASCII using basic character mapping
            ascii_frame = self._image_to_ascii(tmp_path, width, height)

            # Clean up
            os.unlink(tmp_path)

            return ascii_frame

        except Exception as e:
            return [f"Error: {str(e)}"]

    def _image_to_ascii(self, image_path: str, width: int, height: int) -> list[str]:
        """Convert image to ASCII art."""
        try:
            # Try to use PIL if available, otherwise use simple method
            try:
                from PIL import Image

                # Open and resize image
                img = Image.open(image_path)
                img = img.convert("L")  # Convert to grayscale
                img = img.resize((width, height), Image.LANCZOS)

                # ASCII characters from dark to light
                ascii_chars = " .:-=+*#%@"

                ascii_lines = []
                for y in range(height):
                    line = ""
                    for x in range(width):
                        pixel = img.getpixel((x, y))
                        char_index = int((pixel / 255) * (len(ascii_chars) - 1))
                        line += ascii_chars[char_index]
                    ascii_lines.append(line)

                return ascii_lines

            except ImportError:
                # Fallback: use ImageMagick if available
                try:
                    cmd = ["convert", image_path, "-resize", f"{width}x{height}!", "-colorspace", "Gray", "txt:-"]
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        lines = result.stdout.strip().split("\n")[1:]  # Skip header
                        ascii_chars = " .:-=+*#%@"
                        ascii_lines = []

                        for i in range(0, len(lines), width):
                            line = ""
                            for j in range(width):
                                if i + j < len(lines):
                                    # Extract brightness from ImageMagick output
                                    parts = lines[i + j].split()
                                    if len(parts) >= 3:
                                        gray_val = int(parts[2].split(",")[0].replace("(", ""))
                                        char_index = int((gray_val / 255) * (len(ascii_chars) - 1))
                                        line += ascii_chars[char_index]
                                    else:
                                        line += " "
                                else:
                                    line += " "
                            ascii_lines.append(line)
                            if len(ascii_lines) >= height:
                                break

                        return ascii_lines[:height]

                except FileNotFoundError:
                    pass

                # Ultimate fallback
                return ["Video frame conversion requires PIL (pip install Pillow) or ImageMagick"]

        except Exception as e:
            return [f"Error converting image: {str(e)}"]

    def _show_demo_details(self, stdscr, result: DemoResult):
        """Show detailed view of a specific demonstration."""
        scroll = 0

        # Load YAML data
        yaml_data = load_yaml_file(result.yaml_path)

        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()

            # Header
            title = f" {result.demo_name} Details "
            stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
            self.safe_addstr(stdscr, 0, 0, title.center(width)[:width])
            stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            self.safe_addstr(stdscr, 1, 0, "─" * width)

            # Prepare lines to display
            lines = []
            lines.append(("", 0))
            lines.append(("BASIC INFO:", 2))
            lines.append((f"  Demo ID: {result.demo_id}", 0))
            lines.append((f"  Success: {'✓ Yes' if result.success else '✗ No'}", 4 if result.success else 5))
            lines.append((f"  Duration: {result.duration:.2f}s", 6))
            lines.append(
                (f"  Video: {'📹 Available' if result.video_path else '❌ Not found'}", 4 if result.video_path else 5)
            )
            lines.append((f"  YAML File: {result.yaml_path}", 0))
            if result.video_path:
                lines.append((f"  Video File: {result.video_path}", 0))
            lines.append(("", 0))

            if yaml_data:
                # Extract additional information from YAML
                if "timed_language_instructions" in yaml_data and yaml_data["timed_language_instructions"]:
                    instruction = yaml_data["timed_language_instructions"][0].get("language_instruction", "N/A")
                    lines.append(("LANGUAGE INSTRUCTION:", 2))
                    lines.append((f"  {instruction}", 3))
                    lines.append(("", 0))

                if "last_step_reward" in yaml_data:
                    reward = yaml_data["last_step_reward"]
                    lines.append(("FINAL REWARD:", 2))
                    lines.append((f"  {reward:.4f}", 4 if reward > 0 else 5))
                    lines.append(("", 0))

                if "scenario_config" in yaml_data:
                    scenario = yaml_data["scenario_config"]
                    if isinstance(scenario, dict) and "env" in scenario and isinstance(scenario["env"], dict):
                        env_config = scenario["env"]
                        if "skill" in env_config:
                            lines.append(("SKILL:", 2))
                            lines.append((f"  {env_config['skill']}", 3))
                            lines.append(("", 0))

                        if "station_name" in env_config:
                            lines.append(("STATION:", 2))
                            lines.append((f"  {env_config['station_name']}", 3))
                            lines.append(("", 0))

                if "policy_metadata" in yaml_data:
                    policy = yaml_data["policy_metadata"]
                    if isinstance(policy, dict):
                        lines.append(("POLICY INFO:", 2))
                        if "name" in policy:
                            lines.append((f"  Name: {policy['name']}", 0))
                        if "checkpoint_path" in policy:
                            checkpoint = (
                                str(policy["checkpoint_path"]).split("/")[-1] if policy["checkpoint_path"] else "N/A"
                            )
                            lines.append((f"  Checkpoint: {checkpoint}", 0))
                        lines.append(("", 0))

            # Display lines with scrolling
            display_height = height - 4
            max_scroll = max(0, len(lines) - display_height)
            scroll = max(0, min(scroll, max_scroll))

            for idx in range(display_height):
                line_idx = idx + scroll
                if line_idx >= len(lines):
                    break

                line_text, color_pair = lines[line_idx]
                y = idx + 2

                if color_pair > 0:
                    stdscr.attron(curses.color_pair(color_pair))
                self.safe_addstr(stdscr, y, 2, line_text[: width - 4])
                if color_pair > 0:
                    stdscr.attroff(curses.color_pair(color_pair))

            # Footer
            footer_y = height - 2
            self.safe_addstr(stdscr, footer_y, 0, "─" * width)

            scroll_info = ""
            if max_scroll > 0:
                scroll_info = f" ({scroll + 1}/{max_scroll + 1}) "

            controls = f" ↑/↓:Scroll {scroll_info}"
            if result.video_path:
                controls += "| v:Video "
            controls += "| ESC/q:Back "
            stdscr.attron(curses.A_BOLD)
            self.safe_addstr(stdscr, footer_y + 1, 0, controls.center(width)[:width])
            stdscr.attroff(curses.A_BOLD)

            stdscr.refresh()

            # Handle input
            key = stdscr.getch()

            if key in [ord("q"), ord("Q"), 27]:  # q or ESC
                break
            elif key in [curses.KEY_DOWN, ord("j")]:
                if scroll < max_scroll:
                    scroll += 1
            elif key in [curses.KEY_UP, ord("k")]:
                if scroll > 0:
                    scroll -= 1
            elif key == curses.KEY_PPAGE:
                scroll = max(0, scroll - display_height)
            elif key == curses.KEY_NPAGE:
                scroll = min(max_scroll, scroll + display_height)
            elif key in [ord("v"), ord("V")] and result.video_path:
                self._show_video(stdscr, result)

    def _show_help(self, stdscr):
        """Show help screen."""
        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()

            # Header
            stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
            self.safe_addstr(stdscr, 0, 0, " Help ".center(width)[:width])
            stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            self.safe_addstr(stdscr, 1, 0, "─" * width)

            help_text = [
                "",
                "Navigation:",
                "  ↑/↓, j/k     - Move up/down",
                "  Page Up/Down - Page up/down",
                "  Home/End     - Jump to start/end",
                "",
                "Actions:",
                "  Enter        - View demonstration details",
                "  v            - Play video (if available)",
                "  s            - Show summary statistics",
                "  h            - Show duration histogram",
                "  f            - Filter by success/failure (cycles through all/success/failure)",
                "  g            - Toggle grouping (None/Task/Model)",
                "  /            - Search demonstrations by name",
                "  r            - Reset all filters",
                "  ?            - Show this help",
                "  q, ESC       - Quit / Go back",
                "",
                "Video Player Options:",
                "  External players (if available):",
                "    - MPV Player (recommended)",
                "    - VLC Media Player",
                "    - FFmpeg Player (ffplay)",
                "    - MPlayer",
                "  Terminal options:",
                "    - Colored ASCII conversion (good quality)",
                "    - ASCII Art conversion (basic monochrome)",
                "    - Frame extraction to folder",
                "    - Video information display",
                "",
                "ASCII Player Controls (if selected):",
                "  Space        - Play/Pause",
                "  ←/→          - Previous/Next frame",
                "  j/l          - Jump backward/forward 1 second",
                "  ↑/↓          - Increase/Decrease playback speed",
                "  Home/End     - Jump to start/end",
                "  q, ESC       - Exit video player",
                "",
                "Color Legend:",
                "  Green ✓      - Successful demonstrations",
                "  Red ✗        - Failed demonstrations",
                "  📹           - Video available",
                "  Yellow       - Demonstration names",
                "  Magenta      - Duration values",
                "",
                "Summary View:",
                "  Shows overall statistics including success rate",
                "  and duration statistics for all and by success/failure",
                "",
                "Histogram View:",
                "  █ Mostly successful trials (>70% success rate in bin)",
                "  ▒ Mixed results",
                "  ▓ Mostly failed trials (<30% success rate in bin)",
                "",
                "Details View:",
                "  Shows YAML metadata including language instructions,",
                "  rewards, policy information, and skill details",
                "",
                "Video Requirements:",
                "  For external players: install mpv, vlc, ffmpeg, or mplayer",
                "  For ASCII mode: requires ffmpeg and optionally PIL (Pillow)",
                "",
            ]

            for idx, line in enumerate(help_text):
                if idx + 2 < height - 2:
                    if line.endswith(":"):
                        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                        self.safe_addstr(stdscr, idx + 2, 2, line)
                        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
                    else:
                        self.safe_addstr(stdscr, idx + 2, 2, line)

            # Footer
            footer_y = height - 2
            self.safe_addstr(stdscr, footer_y, 0, "─" * width)
            stdscr.attron(curses.A_BOLD)
            self.safe_addstr(stdscr, footer_y + 1, 0, " Press any key to return ".center(width)[:width])
            stdscr.attroff(curses.A_BOLD)

            stdscr.refresh()
            stdscr.getch()
            break


def parse_tasks_file(tasks_file: str) -> list[tuple[str, str, str]]:
    """Parse the tasks file and extract task information."""
    with open(tasks_file) as f:
        content = f.read()

    # Remove f-string prefix
    content = content.replace('f"', '"')
    wrapped_content = f"[{content}]"

    try:
        tasks = ast.literal_eval(wrapped_content)
        # Handle 3-element or 4-element tuples, we care about the task name (2nd element)
        return tasks
    except Exception as e:
        print(f"ERROR: Failed to parse tasks file: {e}", file=sys.stderr)
        return []


def load_yaml_file(yaml_path: str) -> dict[Any, Any]:
    """Load and parse a YAML file with custom tags."""
    try:
        # Create a custom loader that ignores unknown tags
        class CustomLoader(yaml.SafeLoader):
            pass

        def construct_unknown(loader, node):
            """Handle unknown tags by creating a simple object or dict."""
            if isinstance(node, yaml.ScalarNode):
                return loader.construct_scalar(node)
            elif isinstance(node, yaml.SequenceNode):
                return loader.construct_sequence(node)
            elif isinstance(node, yaml.MappingNode):
                return loader.construct_mapping(node)
            else:
                return None

        # Add a default constructor for any unknown tag
        CustomLoader.add_constructor(None, construct_unknown)

        with open(yaml_path) as f:
            return yaml.load(f, Loader=CustomLoader)
    except Exception as e:
        print(f"Error loading {yaml_path}: {e}")
        return {}


def find_video_file(base_folder: str, demo_name: str) -> str | None:
    """Find the corresponding video file for a demonstration."""
    # Extract demo ID from demo_name (e.g., demonstration_1000 -> 1000)
    try:
        demo_id = demo_name.split("_")[-1]

        # Look in vis/ folder for matching video
        vis_folder = os.path.join(base_folder, "vis")
        if not os.path.exists(vis_folder):
            return None

        # Pattern: *-episode_{demo_id}.pkl-mosaic.mp4
        pattern = os.path.join(vis_folder, f"*-episode_{demo_id}.pkl-mosaic.mp4")
        matches = glob.glob(pattern)

        if matches:
            return matches[0]  # Return first match

        return None

    except Exception:
        return None


def _discover_eval_id_paths(checkpoint_base: str, task_name: str) -> list[str]:
    """Discover eval_id subdirectories under evaluation/ on S3.

    Eval campaigns write results to paths like:
        {checkpoint}/evaluation/{subfolder?}/{eval_id}/{task}/rollouts/
    where eval_id looks like "2026-02-19_a1b2c3d4".

    This function lists the evaluation/ prefix on S3 and finds any dated
    eval_id directories — both at the top level and nested inside subfolders —
    then builds candidate paths for the given task.
    Results are sorted most-recent-first so the latest eval is tried first.
    """
    eval_id_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_[0-9a-f]{8}/$")
    paths: list[str] = []

    def _list_s3_prefixes(s3_prefix: str) -> list[str]:
        try:
            result = subprocess.run(
                ["aws", "s3", "ls", s3_prefix],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return []
            prefixes = []
            for line in result.stdout.splitlines():
                if " PRE " in line:
                    token = line.split(" PRE ", 1)[1].strip()
                    prefixes.append(token)
            return prefixes
        except subprocess.TimeoutExpired:
            sys.stderr.write(f"[gather_results] Timed out listing S3 prefix '{s3_prefix}'\n")
            return []
        except OSError as exc:
            sys.stderr.write(f"[gather_results] Failed to invoke AWS CLI for prefix '{s3_prefix}': {exc}\n")
            return []

    def _add_paths_for_eval_ids(base_prefix: str, eval_ids: list[str]):
        for eid in sorted(eval_ids, reverse=True):
            if task_name:
                paths.append(f"{base_prefix}{eid}/{task_name}/rollouts/")
                paths.append(f"{base_prefix}{eid}/{task_name}/summary/")
            paths.append(f"{base_prefix}{eid}/rollouts/")
            paths.append(f"{base_prefix}{eid}/summary/")

    eval_root = f"{checkpoint_base}/evaluation/"
    top_prefixes = _list_s3_prefixes(eval_root)

    # Eval IDs directly under evaluation/
    top_eval_ids = [p.rstrip("/") for p in top_prefixes if eval_id_pattern.match(p)]
    _add_paths_for_eval_ids(eval_root, top_eval_ids)

    # Check non-eval-id subfolders for eval_ids nested inside them.
    # Cap to avoid unbounded S3 list calls on checkpoints with many legacy task directories.
    subfolder_names = [p.rstrip("/") for p in top_prefixes if not eval_id_pattern.match(p)]
    for subfolder in sorted(subfolder_names)[:50]:
        sub_prefix = f"{eval_root}{subfolder}/"
        sub_prefixes = _list_s3_prefixes(sub_prefix)
        sub_eval_ids = [p.rstrip("/") for p in sub_prefixes if eval_id_pattern.match(p)]
        _add_paths_for_eval_ids(sub_prefix, sub_eval_ids)

    return paths


def download_summaries(
    checkpoint_s3_path: str,
    task_name: str,
    output_dir: Path,
    job_name: str | None = None,
    use_cache: bool = True,
) -> str:
    """Download all summary.yaml files for a task."""

    # Ensure checkpoint path ends with / and strip any trailing slashes first to normalize
    checkpoint_s3_path = checkpoint_s3_path.rstrip("/") + "/"

    # Create output directory
    # Use job_name if available to avoid collisions between different runs of same task
    dir_name = job_name if job_name else task_name
    task_dir = output_dir / dir_name

    # Try multiple paths to find the rollouts
    checkpoint_base = checkpoint_s3_path.rstrip("/")
    # Remove checkpoint.ckpt or checkpoint.pt suffix if present
    if checkpoint_base.endswith("/checkpoint.ckpt") or checkpoint_base.endswith("/checkpoint.pt"):
        checkpoint_base = checkpoint_base.rsplit("/", 1)[0]

    # Auto-discover eval_id subdirectories (most recent first)
    paths_to_check = _discover_eval_id_paths(checkpoint_base, task_name)

    # Build a cache key that includes the most recent eval_id path.  Using only
    # checkpoint_s3_path would give a stale cache hit when a *new* eval_id run is
    # written (the checkpoint hasn't changed, but the best source path has).
    cache_key = paths_to_check[0] if paths_to_check else checkpoint_s3_path
    cache_metadata_path = task_dir / ".cache_metadata.txt"

    # Check if cache is valid (same source path was used)
    cache_valid = False
    if (
        use_cache
        and task_dir.exists()
        and list(task_dir.glob("demonstration_*/summary.yaml"))
        and cache_metadata_path.exists()
    ):
        try:
            stored_cache_key = cache_metadata_path.read_text().strip()
            if stored_cache_key == cache_key:
                cache_valid = True
        except Exception:
            pass

    if cache_valid:
        return str(task_dir)

    # If cache is disabled or invalid (different source path), force a clean re-download
    # to avoid mixing stale files with new sync.
    if task_dir.exists() and (not use_cache or not cache_valid):
        shutil.rmtree(task_dir)

    task_dir.mkdir(parents=True, exist_ok=True)

    # Legacy paths (no eval_id)
    if task_name:
        paths_to_check.extend(
            [
                f"{checkpoint_base}/evaluation/{task_name}/rollouts/",
                f"{checkpoint_base}/evaluation/{task_name}/summary/",
            ]
        )
    paths_to_check.extend(
        [
            f"{checkpoint_base}/evaluation/rollouts/",
            f"{checkpoint_base}/evaluation/summary/",
        ]
    )

    found = False
    for s3_base in paths_to_check:
        # Use aws s3 sync to download only summary.yaml files
        # We perform a dry run or listing first? No, sync is efficient if nothing changes
        # But we want to fail fast if path doesn't exist

        # Actually sync
        cmd = ["aws", "s3", "sync", s3_base, str(task_dir), "--exclude", "*", "--include", "*/summary.yaml"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # Check if we got any files
                summary_files = list(task_dir.glob("demonstration_*/summary.yaml"))
                if summary_files:
                    found = True
                    break
        except Exception:
            pass

    # Save the cache key for validation on future runs
    if found:
        with contextlib.suppress(Exception):
            cache_metadata_path.write_text(cache_key)

    if not found:
        # If we didn't find anything, clean up empty dir?
        # Or maybe keep it to show we tried?
        return str(task_dir)

    return str(task_dir)


def find_yaml_files(demo_dir: str) -> list[str]:
    """Find YAML files in the demonstration directory."""
    yaml_files = glob.glob(os.path.join(demo_dir, "*.yaml"))
    if not yaml_files:
        yaml_files = glob.glob(os.path.join(demo_dir, "*.yml"))
    return yaml_files


def process_demonstration_directories(
    base_folder: str, task_name: str | None = None, model_name: str | None = None, use_cache: bool = True
) -> tuple[list[DemoResult], list[str]]:
    """
    Process all demonstration_* directories in the base folder.

    Returns:
        Tuple of (demo_results_list, error_list)
    """
    # Check for cache
    cache_path = os.path.join(base_folder, "cached_results.pkl")
    if use_cache and os.path.exists(cache_path):
        try:
            # Check modification time
            # cache_mtime = os.path.getmtime(cache_path)
            # folder_mtime = os.path.getmtime(base_folder)

            # If cache is newer than folder, use it
            # Note: This might not catch updates to individual files if folder mtime isn't updated
            # A strict check would verify file counts or mtimes, but that defeats the purpose of caching for speed
            # We'll rely on the user to clear cache if needed, or check if we have roughly expected count
            with open(cache_path, "rb") as f:
                results, error_list = pickle.load(f)

            # Update task_name/model_name if it wasn't set in cached results (backwards compatibility)
            # Update task_name/model_name if it wasn't set in cached results (backwards compatibility)
            for r in results:
                if not hasattr(r, "task_name") or (task_name and r.task_name is None):
                    r.task_name = task_name

                if not hasattr(r, "model_name") or (model_name and r.model_name is None):
                    r.model_name = model_name

            return results, error_list
        except Exception:
            # If cache load fails, ignore and re-process
            pass

    demo_pattern = os.path.join(base_folder, "demonstration_*")
    demo_dirs = glob.glob(demo_pattern)

    if not demo_dirs:
        # Don't print this if we are likely recursively searching many folders
        # print(f"No demonstration_* directories found in {base_folder}")
        return [], []

    results = []
    error_list = []

    # Process with progress bar
    # Only show progress bar if there are enough items to matter
    iterator = (
        tqdm(sorted(demo_dirs), desc=f"Processing {os.path.basename(base_folder)}")
        if len(demo_dirs) > 10
        else sorted(demo_dirs)
    )

    for demo_dir in iterator:
        demo_name = os.path.basename(demo_dir)

        # Find YAML files in the directory
        yaml_files = find_yaml_files(demo_dir)

        if not yaml_files:
            error_msg = f"No YAML file found in {demo_name}"
            error_list.append(error_msg)
            continue

        # Prefer summary.yaml if available, otherwise use first YAML file
        yaml_file = None
        for yf in yaml_files:
            if "summary.yaml" in yf:
                yaml_file = yf
                break

        if yaml_file is None:
            yaml_file = yaml_files[0]

        # Load and process the YAML file
        data = load_yaml_file(yaml_file)

        if not data:
            error_msg = f"Failed to load data from {yaml_file}"
            error_list.append(error_msg)
            continue

        # Extract success and last_t fields
        success = data.get("success")
        last_t = data.get("last_t")

        if success is None:
            error_msg = f"'success' field not found in {demo_name}"
            error_list.append(error_msg)
            continue

        if last_t is None:
            error_msg = f"'last_t' field not found in {demo_name}"
            error_list.append(error_msg)
            continue

        result = DemoResult(
            demo_name,
            bool(success),
            float(last_t),
            yaml_file,
            yaml_data=data,
            task_name=task_name,
            model_name=model_name,
        )

        # Find corresponding video file
        video_path = find_video_file(base_folder, demo_name)
        result.video_path = video_path

        results.append(result)

    # Save to cache
    if use_cache and results:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump((results, error_list), f)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    return results, error_list


def print_summary_stats(results: list[DemoResult], error_list: list[str]):
    """Print summary statistics to console."""
    total_trials = len(results)
    successful_trials = sum(1 for r in results if r.success)

    # Check for multiple tasks
    unique_tasks = sorted(list(set(r.task_name for r in results if r.task_name)))
    has_tasks = len(unique_tasks) > 0

    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)

    if total_trials == 0:
        print("No valid trials found!")
        return

    # Per-task breakdown
    if has_tasks:
        print(f"{'Task Name':<50} | {'Success':<10} | {'Total':<8} | {'Rate':<8}")
        print("-" * 86)

        for task in unique_tasks:
            task_results = [r for r in results if r.task_name == task]
            task_success = sum(1 for r in task_results if r.success)
            task_total = len(task_results)
            task_rate = task_success / task_total * 100 if task_total > 0 else 0

            print(f"{task[:48]:<50} | {task_success:<10} | {task_total:<8} | {task_rate:>6.1f}%")

        print("-" * 86)
        print(
            f"{'TOTAL':<50} | {successful_trials:<10} | {total_trials:<8} | "
            f"{successful_trials / total_trials * 100:>6.1f}%"
        )
        print("=" * 80)
        print("")

    # Per-model breakdown
    unique_models = sorted(list(set(r.model_name for r in results if r.model_name)))
    has_models = len(unique_models) > 0

    if has_models:
        print(f"{'Model/Ablation':<50} | {'Success':<10} | {'Total':<8} | {'Rate':<8}")
        print("-" * 86)

        for model in unique_models:
            model_results = [r for r in results if r.model_name == model]
            model_success = sum(1 for r in model_results if r.success)
            model_total = len(model_results)
            model_rate = model_success / model_total * 100 if model_total > 0 else 0

            print(f"{model[:48]:<50} | {model_success:<10} | {model_total:<8} | {model_rate:>6.1f}%")

        print("-" * 86)
        print(
            f"{'TOTAL':<50} | {successful_trials:<10} | {total_trials:<8} | "
            f"{successful_trials / total_trials * 100:>6.1f}%"
        )
        print("=" * 80)
        print("")

    success_rate = successful_trials / total_trials * 100

    print("Overall Statistics:")
    print(f"  Total trials: {total_trials}")
    print(f"  Successful trials: {successful_trials}")
    print(f"  Failed trials: {total_trials - successful_trials}")
    print(f"  Success rate: {success_rate:.2f}%")

    # Video availability
    videos_available = sum(1 for r in results if r.video_path)
    print(f"  Videos available: {videos_available}/{total_trials}")

    if results:
        durations = [r.duration for r in results]
        durations_np = np.array(durations)
        print("\nDuration statistics:")
        print(f"  Mean duration: {durations_np.mean():.2f}s")
        print(f"  Median duration: {np.median(durations_np):.2f}s")
        print(f"  Min duration: {durations_np.min():.2f}s")
        print(f"  Max duration: {durations_np.max():.2f}s")
        print(f"  Std deviation: {durations_np.std():.2f}s")

        # Success vs failure durations
        success_durations = [r.duration for r in results if r.success]
        failure_durations = [r.duration for r in results if not r.success]

        if success_durations:
            print(f"  Mean duration (successful): {np.mean(success_durations):.2f}s")
        if failure_durations:
            print(f"  Mean duration (failed): {np.mean(failure_durations):.2f}s")

    if error_list:
        print(f"\nErrors encountered ({len(error_list)}):")
        # Limit error output if too many
        if len(error_list) > 20:
            for error in error_list[:20]:
                print(f"  - {error}")
            print(f"  ... and {len(error_list) - 20} more errors")
        else:
            for error in error_list:
                print(f"  - {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Gather results from demonstration directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gather_results.py /path/to/rollouts
  python gather_results.py /data2/lbm/rollouts --no-interactive
        """,
    )

    parser.add_argument("folder", type=str, help="Base folder containing demonstration_* directories")

    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive interface, just print summary")

    parser.add_argument("--no-cache", action="store_true", help="Disable reusing cached results")

    args = parser.parse_args()

    # Validate input folder
    # Validate input
    if not os.path.exists(args.folder):
        print(f"Error: Path '{args.folder}' does not exist")
        return 1

    results = []
    error_list = []

    if os.path.isfile(args.folder):
        print(f"Reading task list from: {args.folder}")
        tasks = parse_tasks_file(args.folder)
        print(f"Found {len(tasks)} tasks")

        # Create a temporary directory for downloads if needed, or use a cache dir
        # Let's use a local 'downloaded_results' folder to persist between runs
        cache_dir = Path("downloaded_results")
        cache_dir.mkdir(exist_ok=True)

        for task_info in tqdm(tasks, desc="Processing tasks"):
            # Task tuple format: (job_name, task_name, s3_path, ...)
            if len(task_info) >= 3:
                job_name = task_info[0]
                task_name = task_info[1]
                s3_path = task_info[2]

                # Extract model name from S3 path
                # Pattern: .../ablations/{task_name}/{model_name}/...
                model_name = "Unknown"
                if "/ablations/" in s3_path:
                    try:
                        parts = s3_path.split("/ablations/")
                        if len(parts) > 1:
                            sub_parts = parts[1].split("/")
                            # sub_parts[0] is task_name, sub_parts[1] is model_name
                            if len(sub_parts) > 1:
                                model_name = sub_parts[1]
                    except Exception:
                        pass

                # Fallback: extract the last non-empty path component as model name
                # This handles paths like .../model_checkpoints/vla_diffusion/{run_name}/
                if model_name == "Unknown":
                    try:
                        # Strip trailing slash and split
                        path_parts = s3_path.rstrip("/").split("/")
                        # Get last non-empty component
                        if path_parts:
                            model_name = path_parts[-1]
                    except Exception:
                        pass

                # Check if we have local results first?
                # The prompt implies we should look on S3 if not found or if instructed.
                # Let's try to download/sync from S3
                local_task_dir = download_summaries(
                    s3_path, task_name, cache_dir, job_name=job_name, use_cache=not args.no_cache
                )

                if os.path.isdir(local_task_dir):
                    task_results, task_errors = process_demonstration_directories(
                        local_task_dir, task_name=task_name, model_name=model_name, use_cache=not args.no_cache
                    )
                    results.extend(task_results)
                    error_list.extend(task_errors)
                else:
                    error_list.append(f"Directory not found for task: {task_name} ({job_name})")
            elif len(task_info) >= 2:
                # Fallback for tuples without S3 path if any
                task_name = task_info[1]
                if os.path.isdir(task_name):
                    task_results, task_errors = process_demonstration_directories(
                        task_name, task_name=task_name, use_cache=not args.no_cache
                    )
                    results.extend(task_results)
                    error_list.extend(task_errors)
            else:
                error_list.append(f"Invalid task entry: {str(task_info)}")

    elif os.path.isdir(args.folder):
        print(f"Processing demonstration directories in: {args.folder}")
        # Process all demonstration directories
        results, error_list = process_demonstration_directories(args.folder)
    else:
        print(f"Error: '{args.folder}' is neither a file nor a directory")
        return 1

    print(f"\nProcessed {len(results)} demonstrations")

    if not results:
        print("No valid demonstration results found!")
        return 1

    # Print summary to console
    print_summary_stats(results, error_list)

    # Launch interactive interface unless disabled
    if not args.no_interactive:
        try:
            print("\nLaunching interactive viewer...")
            print("Press '?' for help once in the interface.")
            viewer = DemoResultsViewer(results, args.folder)
            curses.wrapper(viewer.run)
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"\nError in interactive mode: {e}")
            print("Use --no-interactive flag to skip the curses interface")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
