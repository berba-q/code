"""
TITLE:          Function for plotting time series data into PDF files
DESCRIPTION:    Generates multi-page PDF graphical reports with options for line charts, bar charts, stacked charts, and area charts.
VERSION:        v1.0
PURPOSE:        Generate multi-page PDF reports with charts organized by series, plots, pages, and files
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from enum import Enum
import re
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

# Optional imports with fallbacks
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Define class for chart types with enum types
class ChartType(Enum):
    """Supported chart types."""
    LINE = "line"
    BAR = "bar"
    STACKED_BAR = "stacked_bar"
    GROUPED_BAR = "grouped_bar"
    AREA = "area"
    STACKED_AREA = "stacked_area"
    SCATTER = "scatter"
    STEP = "step"
    LINE_SCATTER = "line_scatter"  # Line with markers (default behavior)

# Progress tracking utility class
class ProgressTracker:
    """
    Progress tracking utility that works with or without tqdm.
    Provides consistent progress feedback regardless of environment.
    """
    
    def __init__(
        self, 
        total: int, 
        description: str = "Processing",
        show_progress: bool = True,
        use_tqdm: bool = True,
        callback: Callable[[int, int, str], None] | None = None
    ):
        self.total = total
        self.current = 0
        self.description = description
        self.show_progress = show_progress
        self.callback = callback
        self.start_time = time.time()
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self._pbar = None
        
        if self.show_progress and self.use_tqdm:
            self._pbar = tqdm(
                total=total, 
                desc=description,
                unit="item",
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        elif self.show_progress:
            self._print_progress()
    
    def update(self, n: int = 1, message: str = "") -> None:
        """Update progress by n steps."""
        self.current += n
        
        if self.callback:
            self.callback(self.current, self.total, message)
        
        if self.show_progress:
            if self.use_tqdm and self._pbar:
                self._pbar.update(n)
                if message:
                    self._pbar.set_postfix_str(message[:30])
            else:
                self._print_progress(message)
    
    def _print_progress(self, message: str = "") -> None:
        """Print progress without tqdm."""
        if self.total == 0:
            pct = 100
        else:
            pct = (self.current / self.total) * 100
        
        elapsed = time.time() - self.start_time
        
        if self.current > 0 and self.current < self.total:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = ""
        
        bar_width = 30
        filled = int(bar_width * self.current / max(self.total, 1))
        bar = "█" * filled + "░" * (bar_width - filled)
        
        status = f"\r{self.description}: |{bar}| {self.current}/{self.total} ({pct:.1f}%) {eta_str}"
        if message:
            status += f" - {message[:20]}"
        
        sys.stdout.write(status + " " * 10)
        sys.stdout.flush()
        
        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    
    def close(self) -> None:
        """Close the progress tracker."""
        if self._pbar:
            self._pbar.close()
        elif self.show_progress and self.current < self.total:
            self.current = self.total
            self._print_progress()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ChartStyle:
    """Configuration for chart styling."""
    color_assignment: dict[str, str] = field(default_factory=dict)
    line_assignment: dict[str, str] = field(default_factory=dict)
    symbol_assignment: dict[str, str] = field(default_factory=dict)
    
    # Default matplotlib markers mapped from R's pch values
    DEFAULT_MARKERS: dict[int, str] = field(default_factory=lambda: {
        0: 's', 1: 'o', 2: '^', 3: '+', 4: 'x', 5: 'D',
        6: 'v', 7: 's', 8: '*', 15: 's', 16: 'o', 17: '^',
        18: 'D', 19: 'o', 20: '.', 21: 'o', 22: 's', 23: 'D',
        24: '^', 25: 'v'
    })
    
    # Default line styles
    DEFAULT_LINESTYLES: dict[str, str] = field(default_factory=lambda: {
        'solid': '-', 'dashed': '--', 'dotted': ':', 
        'dashdot': '-.', 'longdash': (0, (5, 5)),
        '1': '-', '2': '--', '3': ':', '4': '-.',
    })
    
    @classmethod
    def from_data(
        cls,
        data: pd.DataFrame,
        series_by: str | None = None,
        symbol_by: str | None = None,
        colors: list[str] | None = None,
        linetypes: list[str] | None = None,
        symbols: list[str | int] | None = None
    ) -> 'ChartStyle':
        """
        Create a ChartStyle by automatically assigning styles based on unique values in data.
        
        Parameters
        ----------
        data : pd.DataFrame
            The dataset to extract unique values from
        series_by : str, optional
            Column name for series (used for color and line assignments)
        symbol_by : str, optional
            Column name for symbols/flags
        colors : list[str], optional
            List of colors to assign. If None, uses default palette.
            Only as many colors as unique series values will be used.
        linetypes : list[str], optional
            List of line types. If None, uses ['solid', 'dashed', 'dotted', 'dashdot'].
            Only as many as unique series values will be used.
        symbols : list[str | int], optional
            List of markers (matplotlib) or R pch values (integers 0-25).
            If None, auto-assigns appropriate markers.
            Only as many as unique flag values will be used.
        
        Returns
        -------
        ChartStyle
            A ChartStyle instance with assignments based on data
        
        """
        color_assignment = {}
        line_assignment = {}
        symbol_assignment = {}
        
        # Map values to matplotlib markers
        pch_to_marker = {
            0: 's', 1: 'o', 2: '^', 3: '+', 4: 'x', 5: 'D',
            6: 'v', 7: 's', 8: '*', 9: 'd', 10: 'o', 11: '*',
            12: 's', 13: 'x', 14: 's', 15: 's', 16: 'o', 17: '^',
            18: 'D', 19: 'o', 20: '.', 21: 'o', 22: 's', 23: 'D',
            24: '^', 25: 'v'
        }
        
        # Default palettes
        default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
        ]
        default_linetypes = ['-', '--', ':', '-.', (0, (5, 5)), (0, (3, 1, 1, 1))]
        default_symbols = ['o', 's', '^', 'D', 'v', 'x', '+', '*', 'p', 'h']
        
        colors = colors or default_colors
        linetypes = linetypes or default_linetypes
        symbols = symbols or default_symbols
        
        # Assign colors and linetypes to series
        if series_by and series_by in data.columns:
            unique_series = data[series_by].dropna().unique()
            for i, val in enumerate(unique_series):
                color_assignment[val] = colors[i % len(colors)]
                line_assignment[val] = linetypes[i % len(linetypes)]
        
        # Assign symbols to flags if symbol_by is provided
        if symbol_by and symbol_by in data.columns:
            unique_symbols = data[symbol_by].dropna().unique()
            for i, val in enumerate(unique_symbols):
                sym = symbols[i % len(symbols)]
                # Convert to matplotlib if it's an integer
                if isinstance(sym, int):
                    sym = pch_to_marker.get(sym, 'o')
                symbol_assignment[str(val)] = sym
        
        return cls(
            color_assignment=color_assignment,
            line_assignment=line_assignment,
            symbol_assignment=symbol_assignment
        )


def create_color_assignment(
    data: pd.DataFrame, 
    column: str, 
    colors: list[str]
) -> dict[str, str]:
    """
    Create color assignment dictionary from data column.
    
    Equivalent to R:
        colour_assignment <- setNames(colours, unique(data[[column]]))
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    column : str
        Column name to get unique values from
    colors : list[str]
        List of colors to assign (hex codes or color names)
    
    Returns
    -------
    dict[str, str]
        Dictionary mapping column values to colors
    --------
    """
    unique_vals = data[column].dropna().unique()
    return {val: colors[i % len(colors)] for i, val in enumerate(unique_vals)}


def create_line_assignment(
    data: pd.DataFrame, 
    column: str, 
    linetypes: list[str] | None = None
) -> dict[str, str]:
    """
    Create line type assignment dictionary from data column.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    column : str
        Column name to get unique values from
    linetypes : list[str], optional
        List of line types. Options: 'solid'/'-', 'dashed'/'--', 'dotted'/':', 'dashdot'/'-.'
        If None, uses ['solid', 'dashed', 'dotted', 'dashdot']
    
    Returns
    -------
    dict[str, str]
        Dictionary mapping column values to line types
    
    --------
    """
    # Map R-style names to matplotlib
    linetype_map = {
        'solid': '-', 'dashed': '--', 'dotted': ':', 'dashdot': '-.',
        'longdash': (0, (5, 5)), 'twodash': (0, (3, 1, 1, 1)),
        '-': '-', '--': '--', ':': ':', '-.': '-.'
    }
    
    if linetypes is None:
        linetypes = ['solid', 'dashed', 'dotted', 'dashdot']
    
    unique_vals = data[column].dropna().unique()
    result = {}
    for i, val in enumerate(unique_vals):
        lt = linetypes[i % len(linetypes)]
        result[val] = linetype_map.get(lt, lt)
    return result


def create_symbol_assignment(
    data: pd.DataFrame, 
    column: str, 
    symbols: list[str | int] | None = None
) -> dict[str, str]:
    """
    Create symbol/marker assignment dictionary from data column.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    column : str
        Column name to get unique values from (e.g., "Flag")
    symbols : list[str | int], optional
        List of matplotlib markers or R pch values (0-25).
        If None, automatically assigns appropriate symbols based on data.
        You can also pass fewer symbols than unique values - they will cycle.
    
    Returns
    -------
    """
    # Map R pch values to matplotlib markers
    pch_to_marker = {
        0: 's', 1: 'o', 2: '^', 3: '+', 4: 'x', 5: 'D',
        6: 'v', 7: 's', 8: '*', 9: 'd', 10: 'o', 11: '*',
        12: 's', 13: 'x', 14: 's', 15: 's', 16: 'o', 17: '^',
        18: 'D', 19: 'o', 20: '.', 21: 'o', 22: 's', 23: 'D',
        24: '^', 25: 'v'
    }
    
    # Default symbols - good variety of distinguishable markers
    default_symbols = ['o', 's', '^', 'D', 'v', 'x', '+', '*', 'p', 'h']
    
    if symbols is None:
        symbols = default_symbols
    
    unique_vals = data[column].dropna().unique()
    result = {}
    for i, val in enumerate(unique_vals):
        sym = symbols[i % len(symbols)]
        # Convert to matplotlib if it's an integer
        if isinstance(sym, int):
            sym = pch_to_marker.get(sym, 'o')
        result[str(val)] = sym
    return result


def get_unique_count(data: pd.DataFrame, column: str) -> int:
    """
    Get the number of unique non-null values in a column.
    
    Useful for determining how many symbols/colors you need.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    column : str
        Column name
    
    Returns
    -------
    int
        Number of unique values
    """
    return data[column].dropna().nunique()


def auto_symbols(data: pd.DataFrame, column: str, use_pch: bool = False) -> list:
    """
    Automatically generate the right number of symbols for a column.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    column : str
        Column name (e.g., "Flag")
    use_pch : bool, optional
        If True, return R pch values (1, 2, 3, ...).
        If False, return matplotlib marker strings ('o', 's', '^', ...).
    
    Returns
    -------
    list
        List of symbols matching the number of unique values
    
    """
    n = data[column].dropna().nunique()
    
    if use_pch:
        pch_values = [1, 2, 3, 4, 5, 6, 15, 16, 17, 18]
        return pch_values[:n]
    else:
        # Matplotlib markers
        markers = ['o', 's', '^', 'D', 'v', 'x', '+', '*', 'p', 'h']
        return markers[:n]


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    output_folder: str | Path = "output/"
    series_by: str = "ItemName"
    x_variable: str = "Year"
    y_variable: str = "Value"
    plots_by: str = "ElementName"
    pages_by: str = "AreaName"
    files_by: str | None = None
    symbol_by: str | None = "Flag"
    units: str | None = "Unit"
    plots_per_row: int = 3
    plots_per_column: int = 2
    file_name: str = "chart"
    
    # Style configuration
    style: ChartStyle = field(default_factory=ChartStyle)
    
    chart_type: ChartType | str = ChartType.LINE_SCATTER
    
    output_format: Literal['pdf', 'png', 'svg', 'all'] = 'pdf'
    dpi: int = 150
    figure_width: float = 29.7 / 1.8  # inches (A4 landscape)
    figure_height: float = 21 / 1.8
    theme: Literal['minimal', 'dark', 'default'] = 'minimal'
    truncate_labels: int = 12
    show_progress: bool = True
    parallel: bool = False
    n_workers: int = 4
    date_format: str | None = None  # For datetime x-axis
    title_fontsize: int = 10
    label_fontsize: int = 9
    legend_fontsize: int = 8
    grid_alpha: float = 0.3
    line_width: float = 1.5
    marker_size: float = 7  # Increased from 4 for better visibility (like R)
    marker_edge_width: float = 1.0  # Edge/border width for markers
    marker_edge_color: str | None = None  # Edge color (None = same as fill)
    bar_width: float = 0.8  # For bar charts
    alpha: float = 0.7  # Transparency for area/bar charts
    
    # Legend configuration
    legend_position: Literal['inside', 'outside', 'bottom', 'right', 'none'] = 'outside'
    show_series_legend: bool = True
    show_symbol_legend: bool = True
    
    # Grid and axis configuration
    show_all_years: bool = True  # Use smart year intervals on x-axis
    show_every_year: bool = False  # Force showing EVERY year label (may overlap for large ranges)
    show_minor_grid: bool = True  # Show minor gridlines (like R)
    show_minor_ticks: bool = False  # Show minor tick marks (usually False for cleaner look)
    minor_grid_alpha: float = 0.15  # Alpha for minor gridlines
    x_axis_interval: int | None = None  # Custom interval for x-axis (None = auto)
    
    # Progress callback
    progress_callback: Callable[[int, int, str], None] | None = None
    
    def __post_init__(self):
        self.output_folder = Path(self.output_folder)
        # Convert string to ChartType if needed
        if isinstance(self.chart_type, str):
            self.chart_type = ChartType(self.chart_type.lower())


@dataclass
class ChartResult:
    """Result object returned after chart generation."""
    files_created: list[Path] = field(default_factory=list)
    pages_generated: int = 0
    plots_generated: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    def summary(self) -> str:
        return (
            f"Charts generated successfully:\n"
            f"  Files created: {len(self.files_created)}\n"
            f"  Total pages: {self.pages_generated}\n"
            f"  Total plots: {self.plots_generated}\n"
            f"  Warnings: {len(self.warnings)}\n"
            f"  Errors: {len(self.errors)}"
        )


def _sanitize_filename(name: str, max_length: int = 15) -> str:
    """Sanitize a string for use in filenames."""
    # Remove non-alphanumeric characters except spaces
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', str(name))
    # Replace spaces with underscores
    sanitized = re.sub(r'\s+', '_', sanitized)
    # Truncate
    return sanitized[:max_length]


def _apply_theme(ax: Axes, config: ChartConfig) -> None:
    """Apply visual theme to axes including grid configuration."""
    theme = config.theme
    
    if theme == 'minimal':
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Major grid
        ax.grid(True, which='major', alpha=config.grid_alpha, linestyle='-', linewidth=0.5)
        # Minor grid (like R's ggplot2)
        if config.show_minor_grid:
            ax.minorticks_on()
            ax.grid(True, which='minor', alpha=config.minor_grid_alpha, linestyle='-', linewidth=0.3)
            # Only show minor grid on x-axis (vertical lines for years)
            ax.tick_params(axis='y', which='minor', left=False)
    elif theme == 'dark':
        ax.set_facecolor('#2E2E2E')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.grid(True, which='major', alpha=0.2, color='white', linestyle='-', linewidth=0.5)
        if config.show_minor_grid:
            ax.minorticks_on()
            ax.grid(True, which='minor', alpha=0.1, color='white', linestyle='-', linewidth=0.3)
            ax.tick_params(axis='y', which='minor', left=False)
    # 'default' uses matplotlib defaults
    elif theme == 'default':
        if config.show_minor_grid:
            ax.minorticks_on()
            ax.grid(True, which='minor', alpha=config.minor_grid_alpha, linestyle='-', linewidth=0.3)


def _format_y_axis(value: float, _: int) -> str:
    """Format y-axis values with space as thousand separator."""
    if abs(value) >= 1e6:
        return f'{value/1e6:.1f}M'
    elif abs(value) >= 1e3:
        return f'{value/1e3:.1f}K'
    return f'{value:,.0f}'.replace(',', ' ')


def _get_color_cycle(n: int, custom_colors: dict[str, str] | None = None) -> list[str]:
    """Generate a color cycle for n series."""
    # Default color palette (colorblind-friendly)
    default_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    
    if custom_colors:
        return list(custom_colors.values())[:n] + default_colors[:(n - len(custom_colors))]
    
    # Cycle through colors if more series than colors
    return [default_colors[i % len(default_colors)] for i in range(n)]


def _create_single_plot(
    ax: Axes,
    data: pd.DataFrame,
    config: ChartConfig,
    plot_title: str,
    page_title: str,
    min_x: float,
    max_x: float,
    series_colors: dict[str, str],
    series_lines: dict[str, str],
    series_markers: dict[str, str]
) -> bool:
    """Create a single plot on the given axes. Returns True if data was plotted."""
    
    # Check if there's any valid data
    valid_data = data.dropna(subset=[config.y_variable])
    
    if valid_data.empty:
        # No data - show message
        ax.text(
            0.5, 0.5,
            f"No data for {config.plots_by}:\n{plot_title}",
            ha='center', va='center',
            transform=ax.transAxes,
            fontsize=config.label_fontsize,
            color='gray'
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return False
    
    # Get unique series
    series_list = data[config.series_by].unique()
    chart_type = config.chart_type
    
    # Prepare data for stacked/grouped charts
    if chart_type in (ChartType.STACKED_BAR, ChartType.GROUPED_BAR, 
                      ChartType.STACKED_AREA):
        # Pivot data for stacked/grouped visualization
        pivot_data = data.pivot_table(
            index=config.x_variable,
            columns=config.series_by,
            values=config.y_variable,
            aggfunc='sum'
        ).fillna(0)
        
        x_vals = pivot_data.index.values
        
        if chart_type == ChartType.STACKED_BAR:
            bottom = np.zeros(len(x_vals))
            bar_width = config.bar_width * 0.8
            
            for series_name in pivot_data.columns:
                y_vals = pivot_data[series_name].values
                color = series_colors.get(series_name, '#1f77b4')
                ax.bar(
                    x_vals, y_vals, bottom=bottom,
                    width=bar_width, label=str(series_name)[:config.truncate_labels],
                    color=color, alpha=config.alpha, edgecolor='white', linewidth=0.5
                )
                bottom += y_vals
        
        elif chart_type == ChartType.GROUPED_BAR:
            n_series = len(pivot_data.columns)
            total_width = config.bar_width * 0.8
            single_width = total_width / n_series
            offsets = np.linspace(
                -total_width/2 + single_width/2,
                total_width/2 - single_width/2,
                n_series
            )
            
            for i, series_name in enumerate(pivot_data.columns):
                y_vals = pivot_data[series_name].values
                color = series_colors.get(series_name, '#1f77b4')
                ax.bar(
                    x_vals + offsets[i], y_vals,
                    width=single_width * 0.9,
                    label=str(series_name)[:config.truncate_labels],
                    color=color, alpha=config.alpha, edgecolor='white', linewidth=0.5
                )
        
        elif chart_type == ChartType.STACKED_AREA:
            y_stack = np.zeros((len(pivot_data.columns), len(x_vals)))
            colors = []
            labels = []
            
            for i, series_name in enumerate(pivot_data.columns):
                y_stack[i] = pivot_data[series_name].values
                colors.append(series_colors.get(series_name, '#1f77b4'))
                labels.append(str(series_name)[:config.truncate_labels])
            
            ax.stackplot(x_vals, y_stack, labels=labels, colors=colors, alpha=config.alpha)
    
    else:
        # Individual series plotting (line, scatter, bar, area, step)
        for series_name in series_list:
            series_data = data[data[config.series_by] == series_name].copy()
            series_data = series_data.dropna(subset=[config.y_variable])
            
            if series_data.empty:
                continue
            
            # Sort by x variable
            series_data = series_data.sort_values(config.x_variable)
            
            x_vals = series_data[config.x_variable].values
            y_vals = series_data[config.y_variable].values
            
            # Get style properties
            color = series_colors.get(series_name, '#1f77b4')
            linestyle = series_lines.get(series_name, '-')
            label = str(series_name)[:config.truncate_labels]
            
            # Plot based on chart type
            if chart_type == ChartType.LINE:
                if len(x_vals) > 1:
                    ax.plot(
                        x_vals, y_vals,
                        color=color, linestyle=linestyle,
                        linewidth=config.line_width, label=label
                    )
                else:
                    edge_color = config.marker_edge_color if config.marker_edge_color else 'white'
                    ax.scatter(x_vals, y_vals, c=color, s=config.marker_size**2, 
                              edgecolors=edge_color, linewidths=config.marker_edge_width,
                              label=label)
            
            elif chart_type == ChartType.LINE_SCATTER:
                # Line with markers - symbols assigned per point based on Flag
                if len(x_vals) > 1:
                    ax.plot(
                        x_vals, y_vals,
                        color=color, linestyle=linestyle,
                        linewidth=config.line_width, label=label
                    )
                
                # Filled markers that support edge colors
                filled_markers = {'o', 's', '^', 'v', '<', '>', 'D', 'd', 'p', 'h', 'H', '8', 'P', 'X'}
                
                # Plot points with symbols based on flag values (like R version)
                if config.symbol_by and config.symbol_by in series_data.columns:
                    # Group points by their flag value and plot each group with its symbol
                    flag_groups = series_data.groupby(config.symbol_by, dropna=False)
                    for flag_val, group_data in flag_groups:
                        flag_key = str(flag_val) if pd.notna(flag_val) else 'NA'
                        marker = series_markers.get(flag_key, 'o')
                        
                        # Only apply edge colors to filled markers
                        if marker in filled_markers:
                            edge_color = config.marker_edge_color if config.marker_edge_color else 'white'
                            ax.scatter(
                                group_data[config.x_variable].values,
                                group_data[config.y_variable].values,
                                c=color, marker=marker,
                                s=config.marker_size ** 2, 
                                edgecolors=edge_color,
                                linewidths=config.marker_edge_width,
                                zorder=5
                            )
                        else:
                            ax.scatter(
                                group_data[config.x_variable].values,
                                group_data[config.y_variable].values,
                                c=color, marker=marker,
                                s=config.marker_size ** 2, 
                                zorder=5
                            )
                else:
                    # No symbol_by column, use default marker
                    edge_color = config.marker_edge_color if config.marker_edge_color else 'white'
                    ax.scatter(
                        x_vals, y_vals,
                        c=color, marker='o',
                        s=config.marker_size ** 2, 
                        edgecolors=edge_color,
                        linewidths=config.marker_edge_width,
                        zorder=5
                    )
            
            elif chart_type == ChartType.SCATTER:
                # Scatter plot with symbols based on flag values
                filled_markers = {'o', 's', '^', 'v', '<', '>', 'D', 'd', 'p', 'h', 'H', '8', 'P', 'X'}
                
                if config.symbol_by and config.symbol_by in series_data.columns:
                    flag_groups = series_data.groupby(config.symbol_by, dropna=False)
                    first_group = True
                    for flag_val, group_data in flag_groups:
                        flag_key = str(flag_val) if pd.notna(flag_val) else 'NA'
                        marker = series_markers.get(flag_key, 'o')
                        
                        if marker in filled_markers:
                            edge_color = config.marker_edge_color if config.marker_edge_color else 'white'
                            ax.scatter(
                                group_data[config.x_variable].values,
                                group_data[config.y_variable].values,
                                c=color, marker=marker,
                                s=config.marker_size ** 2,
                                edgecolors=edge_color,
                                linewidths=config.marker_edge_width,
                                label=label if first_group else None,
                                alpha=config.alpha
                            )
                        else:
                            ax.scatter(
                                group_data[config.x_variable].values,
                                group_data[config.y_variable].values,
                                c=color, marker=marker,
                                s=config.marker_size ** 2,
                                label=label if first_group else None,
                                alpha=config.alpha
                            )
                        first_group = False
                else:
                    edge_color = config.marker_edge_color if config.marker_edge_color else 'white'
                    ax.scatter(
                        x_vals, y_vals,
                        c=color, marker='o',
                        s=config.marker_size ** 2, label=label,
                        alpha=config.alpha
                    )
            
            elif chart_type == ChartType.BAR:
                # Simple bar chart (non-stacked, non-grouped)
                bar_width = config.bar_width / len(series_list)
                series_idx = list(series_list).index(series_name)
                offset = (series_idx - len(series_list)/2 + 0.5) * bar_width
                ax.bar(
                    x_vals + offset, y_vals,
                    width=bar_width * 0.9, label=label,
                    color=color, alpha=config.alpha, edgecolor='white', linewidth=0.5
                )
            
            elif chart_type == ChartType.AREA:
                ax.fill_between(
                    x_vals, y_vals, alpha=config.alpha,
                    color=color, label=label
                )
                ax.plot(x_vals, y_vals, color=color, linewidth=config.line_width * 0.5)
            
            elif chart_type == ChartType.STEP:
                ax.step(
                    x_vals, y_vals, where='mid',
                    color=color, linestyle=linestyle,
                    linewidth=config.line_width, label=label
                )
    
    # Configure axes
    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    
    # Set y-axis to start from 0 (or min if negative)
    y_min = min(0, valid_data[config.y_variable].min())
    y_max = valid_data[config.y_variable].max()
    
    if y_max == y_min:
        if y_max == 0:
            y_padding = 1  # all zero show range 0-1
        else:
            y_padding = abs(y_max) * 0.1  # 10% padding
    else:
        y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min, y_max + y_padding)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_format_y_axis))
    
    # Handle x-axis (integer years or dates)
    if config.date_format:
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter(config.date_format))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        # Assume numeric x-axis (years)
        from matplotlib.ticker import MultipleLocator, FixedLocator
        
        x_range = max_x - min_x
        start_year = int(min_x)
        end_year = int(max_x)
        
        if config.show_every_year:
            # Show year label (may overlap for large ranges)
            major_ticks = list(range(start_year, end_year + 1))
            ax.xaxis.set_major_locator(FixedLocator(major_ticks))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
            # Rotate labels if too many years
            if x_range > 15:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=config.label_fontsize - 2)
        
        elif config.show_all_years:
            # Smart intervals based on range
            if config.x_axis_interval:
                interval = config.x_axis_interval
            elif x_range <= 10:
                interval = 1  # Show every year
            elif x_range <= 15:
                interval = 2  # Show every 2 years
            elif x_range <= 25:
                interval = 4  # Show every 4 years  
            elif x_range <= 40:
                interval = 5  # Show every 5 years
            else:
                interval = 10
            
            # Generate tick positions - start from first year, end at last year
            major_ticks = list(range(start_year, end_year + 1, interval))
            # Make sure we include the last year
            if end_year not in major_ticks:
                major_ticks.append(end_year)
            
            ax.xaxis.set_major_locator(FixedLocator(major_ticks))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
            
            # Minor ticks for gridlines only (not for tick marks)
            if config.show_minor_grid and interval > 1:
                ax.xaxis.set_minor_locator(MultipleLocator(1))
            
            # Control minor tick mark visibility
            if not config.show_minor_ticks:
                ax.tick_params(axis='x', which='minor', length=0)
        else:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    
    # Labels and title
    full_title = f"{page_title}\n{plot_title}"
    ax.set_title(full_title, fontsize=config.title_fontsize, pad=10)
    ax.set_xlabel(config.x_variable, fontsize=config.label_fontsize)
    
    # Y-axis label with units
    y_label = config.y_variable
    if config.units and config.units in data.columns:
        unit_vals = data[config.units].dropna().unique()
        if len(unit_vals) > 0:
            y_label = f"{config.y_variable} ({unit_vals[0]})"
    ax.set_ylabel(y_label, fontsize=config.label_fontsize)
    
    # Create legends based on configuration
    if config.legend_position == 'none':
        # No legends
        pass
    else:
        # Collect all legend handles and labels
        series_handles = []
        series_labels = []
        symbol_handles = []
        symbol_labels = []
        
        # Get series legend handles from existing plot
        if config.show_series_legend and len(series_list) > 1:
            handles, labels = ax.get_legend_handles_labels()
            # Filter to unique labels (series names)
            seen = set()
            for h, l in zip(handles, labels):
                if l not in seen:
                    series_handles.append(h)
                    series_labels.append(l)
                    seen.add(l)
        
        # Create symbol legend handles - show ALL flags present in THIS plot
        if config.show_symbol_legend and config.symbol_by and config.symbol_by in data.columns:
            # Get all unique flag values from this plot's data
            symbol_vals = data[config.symbol_by].dropna().unique()
            if len(symbol_vals) > 0:
                filled_markers = {'o', 's', '^', 'v', '<', '>', 'D', 'd', 'p', 'h', 'H', '8', 'P', 'X'}
                for sym_val in sorted(symbol_vals, key=str):
                    sym_key = str(sym_val)
                    marker = series_markers.get(sym_key, 'o')
                    
                    if marker in filled_markers:
                        edge_color = config.marker_edge_color if config.marker_edge_color else 'white'
                        handle = ax.scatter([], [], marker=marker, c='gray', 
                                           s=config.marker_size**2, 
                                           edgecolors=edge_color,
                                           linewidths=config.marker_edge_width,
                                           label=sym_key)
                    else:
                        handle = ax.scatter([], [], marker=marker, c='gray', 
                                           s=config.marker_size**2, 
                                           label=sym_key)
                    symbol_handles.append(handle)
                    symbol_labels.append(sym_key)
        
        # Position legends based on config
        if config.legend_position == 'outside' or config.legend_position == 'bottom':
            # Place legends below the plot - won't obstruct data
            if series_handles:
                series_legend = ax.legend(
                    series_handles, series_labels,
                    loc='upper center',
                    bbox_to_anchor=(0.3, -0.15),  # Below plot, left side
                    fontsize=config.legend_fontsize,
                    framealpha=0.9,
                    ncol=min(3, len(series_handles)),
                    title=config.series_by[:12],
                    title_fontsize=config.legend_fontsize
                )
                ax.add_artist(series_legend)
            
            if symbol_handles:
                ax.legend(
                    symbol_handles, symbol_labels,
                    loc='upper center',
                    bbox_to_anchor=(0.75, -0.15),  # Below plot, right side
                    fontsize=config.legend_fontsize,
                    framealpha=0.9,
                    ncol=min(4, len(symbol_handles)),
                    title=config.symbol_by[:12] if config.symbol_by else "Symbol",
                    title_fontsize=config.legend_fontsize
                )
        
        elif config.legend_position == 'right':
            # Place legends to the right of the plot
            if series_handles:
                series_legend = ax.legend(
                    series_handles, series_labels,
                    loc='upper left',
                    bbox_to_anchor=(1.02, 1.0),  # Right of plot, top
                    fontsize=config.legend_fontsize,
                    framealpha=0.9,
                    ncol=1,
                    title=config.series_by[:12],
                    title_fontsize=config.legend_fontsize
                )
                ax.add_artist(series_legend)
            
            if symbol_handles:
                ax.legend(
                    symbol_handles, symbol_labels,
                    loc='lower left',
                    bbox_to_anchor=(1.02, 0.0),  # Right of plot, bottom
                    fontsize=config.legend_fontsize,
                    framealpha=0.9,
                    ncol=1,
                    title=config.symbol_by[:12] if config.symbol_by else "Symbol",
                    title_fontsize=config.legend_fontsize
                )
        
        elif config.legend_position == 'inside':
            # Place legends inside plot (original behavior) - use best location
            if series_handles:
                series_legend = ax.legend(
                    series_handles, series_labels,
                    loc='best',  # Let matplotlib find best spot
                    fontsize=config.legend_fontsize,
                    framealpha=0.9,
                    ncol=min(2, len(series_handles)),
                    title=config.series_by[:10],
                    title_fontsize=config.legend_fontsize
                )
                ax.add_artist(series_legend)
            
            if symbol_handles:
                # Try to place symbol legend in a different corner
                ax.legend(
                    symbol_handles, symbol_labels,
                    loc='lower right',
                    fontsize=config.legend_fontsize,
                    framealpha=0.9,
                    ncol=min(2, len(symbol_handles)),
                    title=config.symbol_by[:10] if config.symbol_by else "Symbol",
                    title_fontsize=config.legend_fontsize
                )
    
    # Apply theme
    _apply_theme(ax, config)
    
    return True


def _prepare_data(data: pd.DataFrame, config: ChartConfig) -> pd.DataFrame:
    """Prepare and validate the data for plotting."""
    
    # Determine columns to select
    columns = [config.series_by, config.x_variable, config.y_variable, 
               config.plots_by, config.pages_by]
    
    if config.files_by:
        columns.append(config.files_by)
    if config.symbol_by and config.symbol_by in data.columns:
        columns.append(config.symbol_by)
    if config.units and config.units in data.columns:
        columns.append(config.units)
    
    # Remove duplicates and filter to existing columns
    columns = list(dict.fromkeys(col for col in columns if col in data.columns))
    
    # Select relevant columns
    df = data[columns].copy()
    
    # Check for duplicates
    if df.duplicated().any():
        n_dups = df.duplicated().sum()
        logger.warning(f"Found {n_dups} duplicate rows. Removing duplicates.")
        df = df.drop_duplicates()
    
    # Truncate series labels if they are strings and truncation is unambiguous
    if df[config.series_by].dtype == object:
        original_unique = df[config.series_by].nunique()
        df['_series_truncated'] = df[config.series_by].astype(str).str[:config.truncate_labels]
        truncated_unique = df['_series_truncated'].nunique()
        
        if original_unique == truncated_unique:
            df[config.series_by] = df['_series_truncated']
        df = df.drop(columns=['_series_truncated'])
    
    return df


def _build_style_mappings(
    data: pd.DataFrame, 
    config: ChartConfig
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Build color, line, and marker style mappings for all series."""
    
    series_list = data[config.series_by].unique()
    n_series = len(series_list)
    
    # Colors
    if config.style.color_assignment:
        colors = {s: config.style.color_assignment.get(s, '#1f77b4') for s in series_list}
    else:
        color_cycle = _get_color_cycle(n_series)
        colors = dict(zip(series_list, color_cycle))
    
    # Line styles
    default_lines = ['-', '--', ':', '-.']
    if config.style.line_assignment:
        lines = {s: config.style.line_assignment.get(s, '-') for s in series_list}
    else:
        lines = {s: default_lines[i % len(default_lines)] for i, s in enumerate(series_list)}
    
    # Markers (for symbol_by values)
    markers = {}
    if config.symbol_by and config.symbol_by in data.columns:
        symbol_values = data[config.symbol_by].dropna().unique()
        default_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        if config.style.symbol_assignment:
            markers = {str(s): config.style.symbol_assignment.get(str(s), 'o') for s in symbol_values}
        else:
            markers = {str(s): default_markers[i % len(default_markers)] 
                      for i, s in enumerate(symbol_values)}
    
    return colors, lines, markers


def multiple_line_charts(
    data: pd.DataFrame,
    config: ChartConfig | None = None,
    **kwargs
) -> ChartResult:
    """
    Generate multi-page PDF charts from time series data.
    
    This function creates publication-ready charts organized hierarchically:
    - Multiple files (optional, split by `files_by`)
    - Multiple pages per file (split by `pages_by`)
    - Multiple plots per page (split by `plots_by`)
    - Multiple series per plot (split by `series_by`)
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data containing the time series values
    config : ChartConfig, optional
        Configuration object. If not provided, a default config is created
        with any kwargs passed to this function.
    **kwargs
        Additional keyword arguments passed to ChartConfig if config is None
    
    Returns
    -------
    ChartResult
        Object containing information about generated files, pages, and any issues
    """
    
    # Initialize config
    if config is None:
        config = ChartConfig(**kwargs)
    
    result = ChartResult()
    
    # Validate output folder
    if not config.output_folder.exists(): #type: ignore
        try:
            config.output_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output folder: {config.output_folder}")
        except Exception as e:
            result.errors.append(f"Cannot create output folder: {e}")
            return result
    
    # Prepare data
    try:
        df = _prepare_data(data, config)
    except Exception as e:
        result.errors.append(f"Data preparation failed: {e}")
        return result
    
    # Get global x-axis limits
    min_x = df[config.x_variable].min()
    max_x = df[config.x_variable].max()
    
    # Build style mappings
    series_colors, series_lines, series_markers = _build_style_mappings(df, config)
    
    # Determine file splits
    if config.files_by and config.files_by in df.columns:
        file_groups = df[config.files_by].unique()
    else:
        file_groups = [None]
    
    # Calculate total work for progress tracking
    total_pages = 0
    for file_group in file_groups:
        if file_group is not None:
            file_data = df[df[config.files_by] == file_group]
        else:
            file_data = df
        
        plots_per_page = config.plots_per_row * config.plots_per_column
        page_plot_combos = file_data[[config.pages_by, config.plots_by]].drop_duplicates()
        
        for page_name in page_plot_combos[config.pages_by].unique():
            page_plots = page_plot_combos[page_plot_combos[config.pages_by] == page_name][config.plots_by].tolist()
            total_pages += (len(page_plots) + plots_per_page - 1) // plots_per_page
    
    # Initialize main progress tracker
    logger.info(f"Starting chart generation: {len(file_groups)} file(s), ~{total_pages} page(s)")
    
    file_progress = ProgressTracker(
        total=len(file_groups),
        description="Files",
        show_progress=config.show_progress,
        callback=config.progress_callback
    )
    
    for file_group in file_groups:
        # Filter data for this file
        if file_group is not None:
            file_data = df[df[config.files_by] == file_group].copy()
            file_suffix = _sanitize_filename(str(file_group))
        else:
            file_data = df.copy()
            # Use first plots_by value for naming if no files_by
            first_plot = str(file_data[config.plots_by].iloc[0])
            file_suffix = _sanitize_filename(first_plot)
        
        # Generate output filename
        output_path = config.output_folder / f"Plot_{config.file_name}_{file_suffix}.pdf"
        
        # Calculate page assignments
        plots_per_page = config.plots_per_row * config.plots_per_column
        
        # Get unique combinations of pages_by and plots_by
        page_plot_combos = file_data[[config.pages_by, config.plots_by]].drop_duplicates()
        
        # Assign plots to pages
        page_plot_combos = page_plot_combos.sort_values([config.pages_by, config.plots_by])
        
        # Group by pages_by first, then split into sub-pages if needed
        pages_list = []
        for page_name in page_plot_combos[config.pages_by].unique():
            page_plots = page_plot_combos[page_plot_combos[config.pages_by] == page_name][config.plots_by].tolist()
            
            # Split into chunks of plots_per_page
            for i in range(0, len(page_plots), plots_per_page):
                chunk = page_plots[i:i + plots_per_page]
                pages_list.append((page_name, chunk))
        
        # Create PDF
        try:
            with PdfPages(output_path) as pdf:
                # Apply dark theme to figure if needed
                if config.theme == 'dark':
                    plt.style.use('dark_background')
                else:
                    plt.style.use('default')
                
                # Page progress tracker
                page_progress = ProgressTracker(
                    total=len(pages_list),
                    description=f"  Pages ({file_suffix[:15]})",
                    show_progress=config.show_progress,
                    callback=config.progress_callback
                )
                
                for page_idx, (page_name, plot_list) in enumerate(pages_list):
                    # Create figure with grid layout
                    fig = plt.figure(figsize=(config.figure_width, config.figure_height), dpi=config.dpi)
                    
                    if config.theme == 'dark':
                        fig.patch.set_facecolor('#1E1E1E')
                    
                    # Adjust spacing based on legend position
                    if config.legend_position in ('outside', 'bottom'):
                        # More vertical space for legends below plots
                        hspace = 0.5
                        wspace = 0.35
                        bottom_margin = 0.12
                    elif config.legend_position == 'right':
                        # More horizontal space for legends on right
                        hspace = 0.4
                        wspace = 0.45
                        bottom_margin = 0.08
                    else:
                        # Inside or none - default spacing
                        hspace = 0.4
                        wspace = 0.3
                        bottom_margin = 0.08
                    
                    gs = gridspec.GridSpec(
                        config.plots_per_column, 
                        config.plots_per_row,
                        figure=fig,
                        hspace=hspace,
                        wspace=wspace,
                        bottom=bottom_margin,
                        top=0.92,
                        left=0.08,
                        right=0.95 if config.legend_position != 'right' else 0.85
                    )
                    
                    # Create each plot
                    for idx, plot_name in enumerate(plot_list):
                        row = idx // config.plots_per_row
                        col = idx % config.plots_per_row
                        
                        ax = fig.add_subplot(gs[row, col])
                        
                        # Get data for this plot
                        plot_data = file_data[
                            (file_data[config.pages_by] == page_name) &
                            (file_data[config.plots_by] == plot_name)
                        ]
                        
                        # Create the plot
                        had_data = _create_single_plot(
                            ax, plot_data, config,
                            str(plot_name), str(page_name),
                            min_x, max_x,
                            series_colors, series_lines, series_markers
                        )
                        
                        result.plots_generated += 1
                        if not had_data:
                            result.warnings.append(f"No data for {page_name}/{plot_name}")
                    
                    # Fill empty slots
                    for idx in range(len(plot_list), plots_per_page):
                        row = idx // config.plots_per_row
                        col = idx % config.plots_per_row
                        ax = fig.add_subplot(gs[row, col])
                        ax.axis('off')
                    
                    # Save page to PDF
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    
                    result.pages_generated += 1
                    page_progress.update(1, f"Page {page_idx + 1}/{len(pages_list)}")
                
                page_progress.close()
                
                # Reset style
                plt.style.use('default')
            
            result.files_created.append(output_path)
            logger.info(f"Created: {output_path}")
            file_progress.update(1, str(output_path.name)[:20])
            
            # Also save as other formats if requested
            if config.output_format in ('png', 'all'):
                _save_as_images(file_data, config, output_path.stem, 'png', 
                              min_x, max_x, series_colors, series_lines, series_markers,
                              pages_list)
            
            if config.output_format in ('svg', 'all'):
                _save_as_images(file_data, config, output_path.stem, 'svg',
                              min_x, max_x, series_colors, series_lines, series_markers,
                              pages_list)
                
        except Exception as e:
            result.errors.append(f"Error generating {output_path}: {e}")
            logger.error(f"Error generating {output_path}: {e}")
            file_progress.update(1, f"Error: {str(e)[:15]}")
    
    file_progress.close()
    
    # Print summary
    if config.show_progress:
        print(f"\n{result.summary()}")
    
    return result


def _save_as_images(
    file_data: pd.DataFrame,
    config: ChartConfig,
    base_name: str,
    fmt: str,
    min_x: float,
    max_x: float,
    series_colors: dict,
    series_lines: dict,
    series_markers: dict,
    pages_list: list
) -> None:
    """Save pages as individual image files."""
    
    img_folder = config.output_folder / f"{base_name}_{fmt}"
    img_folder.mkdir(exist_ok=True)
    
    for page_idx, (page_name, plot_list) in enumerate(pages_list):
        fig = plt.figure(figsize=(config.figure_width, config.figure_height), dpi=config.dpi)
        
        gs = gridspec.GridSpec(
            config.plots_per_column,
            config.plots_per_row,
            figure=fig,
            hspace=0.4,
            wspace=0.3
        )
        
        plots_per_page = config.plots_per_row * config.plots_per_column
        
        for idx, plot_name in enumerate(plot_list):
            row = idx // config.plots_per_row
            col = idx % config.plots_per_row
            
            ax = fig.add_subplot(gs[row, col])
            
            plot_data = file_data[
                (file_data[config.pages_by] == page_name) &
                (file_data[config.plots_by] == plot_name)
            ]
            
            _create_single_plot(
                ax, plot_data, config,
                str(plot_name), str(page_name),
                min_x, max_x,
                series_colors, series_lines, series_markers
            )
        
        # Fill empty slots
        for idx in range(len(plot_list), plots_per_page):
            row = idx // config.plots_per_row
            col = idx % config.plots_per_row
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
        
        page_path = img_folder / f"page_{page_idx + 1:03d}.{fmt}"
        fig.savefig(page_path, bbox_inches='tight', dpi=config.dpi)
        plt.close(fig)


# Convenience function for quick usage
def quick_plot(
    data: pd.DataFrame,
    x: str = "Year",
    y: str = "Value",
    series: str = "ItemName",
    output: str = "output/quick_chart.pdf",
    **kwargs
) -> ChartResult:
    """
    Quick single-file chart generation with minimal configuration.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis
    series : str
        Column name for grouping into series
    output : str
        Output file path
    **kwargs
        Additional arguments passed to ChartConfig
    
    Returns
    -------
    ChartResult
    """
    output_path = Path(output)
    
    config = ChartConfig(
        output_folder=str(output_path.parent),
        x_variable=x,
        y_variable=y,
        series_by=series,
        plots_by=series,  # One plot showing all series
        pages_by=series,  # Single page
        file_name=output_path.stem,
        plots_per_row=1,
        plots_per_column=1,
        **kwargs
    )
    
    return multiple_line_charts(data, config)