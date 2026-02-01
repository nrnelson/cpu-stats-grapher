#!/usr/bin/env python3
"""
CPU Temperature Analysis Tool

Author: Nathan Nelson <nrnelson@gmail.com>
Source: https://github.com/nrnelson/cpu-stats-grapher

Analyzes CPU temperature logs and generates visualizations with statistical
analysis. Designed to work with data collected by cpu-stats-logger.sh.

Features:
- Temperature timeline with min/max envelope
- Clock speed analysis with throttling detection
- Power consumption tracking (socket + CCD breakdown)
- Temperature vs CPU usage correlation
- Temperature distribution histogram
- CPU usage timeline (fallback when no temperature data)

Graceful Degradation:
- Detects available data columns and generates only applicable graphs
- Falls back to numpy for correlation if scipy unavailable
- Generates usage timeline if no temperature data available

Usage:
    python cpu-stats-grapher.py logfile.log [OPTIONS]

See --help for all options.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional scipy import for correlation analysis
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze CPU temperature logs and generate visualizations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s cpu_temps_compile.log
  %(prog)s cpu_temps_compile.log -o my_analysis --resample 5s
  %(prog)s cpu_temps_compile.log --tjmax 90 --no-show
        '''
    )
    parser.add_argument('input', type=Path, help='Path to the log file')
    parser.add_argument('-o', '--output', default='cpu_analysis',
                        help='Base name for output files (default: cpu_analysis)')
    parser.add_argument('--resample', default='10s',
                        help='Resample interval for smoothing (default: 10s)')
    parser.add_argument('--tjmax', type=float, default=95.0,
                        help='Thermal junction max temperature (default: 95)')
    parser.add_argument('--no-show', action='store_true',
                        help='Skip displaying plots (for headless use)')
    parser.add_argument('--open', action='store_true',
                        help='Open generated images with default viewer')
    parser.add_argument('-f', '--fahrenheit', action='store_true',
                        help='Display temperatures in Fahrenheit')
    return parser.parse_args()


def c_to_f(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def detect_available_data(df: pd.DataFrame) -> dict:
    """
    Detect which data columns are available and have sufficient valid data.

    Returns dict with boolean flags for each data type.
    """
    available = {
        'temperature': False,
        'usage': False,
        'power': False,
        'power_ccd': False,
        'clocks': False,
    }

    # Check temperature (need >10% valid data)
    if 'temp_c' in df.columns:
        temp = pd.to_numeric(df['temp_c'], errors='coerce')
        valid_ratio = temp.notna().sum() / len(df)
        available['temperature'] = valid_ratio > 0.1

    # Check usage
    if 'usage_pct' in df.columns:
        usage = df['usage_pct']
        if pd.api.types.is_string_dtype(usage):
            usage = usage.str.rstrip('%')
        usage = pd.to_numeric(usage, errors='coerce')
        valid_ratio = usage.notna().sum() / len(df)
        available['usage'] = valid_ratio > 0.1

    # Check power
    if 'power_w' in df.columns:
        power = pd.to_numeric(df['power_w'], errors='coerce')
        valid_ratio = power.notna().sum() / len(df)
        available['power'] = valid_ratio > 0.1

    # Check CCD power breakdown
    if 'ccd0_w' in df.columns and 'ccd1_w' in df.columns:
        ccd0 = pd.to_numeric(df['ccd0_w'], errors='coerce')
        ccd1 = pd.to_numeric(df['ccd1_w'], errors='coerce')
        # Need at least one CCD with valid data
        available['power_ccd'] = (ccd0.notna().sum() / len(df) > 0.1 or
                                   ccd1.notna().sum() / len(df) > 0.1)

    # Check clocks (any core*_mhz columns with valid data)
    core_cols = [c for c in df.columns if c.startswith('core') and c.endswith('_mhz')]
    if core_cols:
        # Check if at least one core has valid data
        for col in core_cols:
            clocks = pd.to_numeric(df[col], errors='coerce')
            if clocks.notna().sum() / len(df) > 0.1:
                available['clocks'] = True
                break

    return available


def load_data(filepath: Path) -> pd.DataFrame:
    """Load and clean the temperature log file."""
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(filepath, sep='\t', engine='python')
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Rename columns for easier access
    df.columns = df.columns.str.strip()

    # Build datetime index from date + time columns
    # Header format: date, time, load_avg, usage_pct, [temp_c], [power_w, ccd0_w, ccd1_w], [core*_mhz]
    first_col = df.columns[0].lower()

    # Parse datetime - date and time are separate columns
    if first_col == 'date' or len(df.columns) >= 4:
        # Re-read with whitespace separator to handle tab-separated data
        df = pd.read_csv(filepath, sep=r'\s+', engine='python', skiprows=1, header=None)

        # Get header to determine column layout
        with open(filepath, 'r') as f:
            header_line = f.readline().strip()
        header_cols = header_line.split('\t')

        num_data_cols = len(df.columns)

        # Header has: date, time, load_avg, usage_pct, [optional cols...]
        # Use header directly since it now matches the data format
        if len(header_cols) == num_data_cols:
            df.columns = header_cols
        else:
            # Fallback: build column names from header, accounting for any mismatch
            core_cols = [f'core{i}_mhz' for i in range(64)]

            # Check various formats by column count
            if num_data_cols >= 40:
                # Full format with temp, power, CCD, and clocks
                df.columns = ['date', 'time', 'load_avg', 'usage_pct', 'temp_c',
                              'power_w', 'ccd0_w', 'ccd1_w'] + core_cols[:num_data_cols - 8]
            elif num_data_cols >= 36:
                # Temp + clocks (no power)
                df.columns = ['date', 'time', 'load_avg', 'usage_pct', 'temp_c'] + core_cols[:num_data_cols - 5]
            elif 'temp_c' in header_cols:
                # Has temperature - build based on header
                base_cols = ['date', 'time', 'load_avg', 'usage_pct', 'temp_c']
                if 'power_w' in header_cols:
                    if 'ccd0_w' in header_cols:
                        base_cols.extend(['power_w', 'ccd0_w', 'ccd1_w'])
                    else:
                        base_cols.append('power_w')
                remaining = num_data_cols - len(base_cols)
                df.columns = base_cols + core_cols[:remaining]
            else:
                # Minimal format: date, time, load_avg, usage_pct + optional columns
                base_cols = ['date', 'time', 'load_avg', 'usage_pct']
                remaining = num_data_cols - len(base_cols)
                df.columns = base_cols + core_cols[:remaining]

        # Combine date and time into datetime
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S')
    else:
        df['datetime'] = pd.to_datetime(df.iloc[:, 0])

    df.set_index('datetime', inplace=True)

    # Clean numeric columns
    if 'temp_c' in df.columns:
        df['temp_c'] = pd.to_numeric(df['temp_c'], errors='coerce')

    if 'load_avg' in df.columns:
        df['load_avg'] = pd.to_numeric(df['load_avg'], errors='coerce')

    # Strip % from usage_pct and convert to numeric
    if 'usage_pct' in df.columns:
        if pd.api.types.is_string_dtype(df['usage_pct']):
            df['usage_pct'] = df['usage_pct'].str.rstrip('%').astype(float)
        else:
            df['usage_pct'] = pd.to_numeric(df['usage_pct'], errors='coerce')

    # Parse power columns if present (convert "-" to NaN)
    for power_col in ['power_w', 'ccd0_w', 'ccd1_w']:
        if power_col in df.columns:
            df[power_col] = pd.to_numeric(df[power_col], errors='coerce')

    # Convert core frequencies to numeric
    core_cols = [c for c in df.columns if c.startswith('core') and c.endswith('_mhz')]
    for col in core_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate average clock speed across all cores
    if core_cols:
        df['avg_clock_mhz'] = df[core_cols].mean(axis=1)

    return df


def compute_statistics(df: pd.DataFrame, tjmax: float, use_fahrenheit: bool = False,
                       available: dict | None = None) -> dict:
    """Compute comprehensive statistics from the data."""
    if available is None:
        available = detect_available_data(df)

    duration = (df.index[-1] - df.index[0]).total_seconds()

    stats_dict = {
        'duration_min': duration / 60,
        'samples': len(df),
    }

    # Temperature stats (only if available)
    if available['temperature']:
        temp = df['temp_c'].dropna()

        # Thresholds in Celsius for calculations
        thresh_85 = 85
        thresh_90 = 90

        # Calculate stats in Celsius first
        temp_min = temp.min()
        temp_max = temp.max()
        temp_mean = temp.mean()
        temp_std = temp.std()
        temp_median = temp.median()
        headroom = tjmax - temp_max

        # Convert to Fahrenheit if requested
        if use_fahrenheit:
            temp_min = c_to_f(temp_min)
            temp_max = c_to_f(temp_max)
            temp_mean = c_to_f(temp_mean)
            temp_std = temp_std * 9 / 5  # Std dev scales differently
            temp_median = c_to_f(temp_median)
            headroom = headroom * 9 / 5  # Delta scales by ratio only

        stats_dict.update({
            'temp_min': temp_min,
            'temp_max': temp_max,
            'temp_mean': temp_mean,
            'temp_std': temp_std,
            'temp_median': temp_median,
            'time_above_85': (df['temp_c'].dropna() > thresh_85).sum() / len(temp) * 100,
            'time_above_90': (df['temp_c'].dropna() > thresh_90).sum() / len(temp) * 100,
            'time_above_tjmax': (df['temp_c'].dropna() > tjmax).sum() / len(temp) * 100,
            'headroom': headroom,
        })

    # Load/Usage stats
    if available['usage']:
        usage = df['usage_pct'].dropna()
        stats_dict['usage_mean'] = usage.mean()
        stats_dict['usage_max'] = usage.max()
        stats_dict['usage_min'] = usage.min()

    # Load average stats
    if 'load_avg' in df.columns:
        load = df['load_avg'].dropna()
        if len(load) > 0:
            stats_dict['load_mean'] = load.mean()
            stats_dict['load_max'] = load.max()

    # Clock speed stats
    if available['clocks'] and 'avg_clock_mhz' in df.columns:
        clocks = df['avg_clock_mhz'].dropna()
        stats_dict['clock_mean'] = clocks.mean()
        stats_dict['clock_min'] = clocks.min()
        stats_dict['clock_max'] = clocks.max()
        stats_dict['clock_std'] = clocks.std()

    # Correlation between temp and usage (only if both available)
    if available['temperature'] and available['usage']:
        valid = df[['temp_c', 'usage_pct']].dropna()
        if len(valid) > 2:
            if SCIPY_AVAILABLE:
                r, p = scipy_stats.pearsonr(valid['temp_c'], valid['usage_pct'])
                stats_dict['temp_usage_corr'] = r
                stats_dict['temp_usage_pval'] = p
            else:
                # Numpy fallback for correlation
                r = np.corrcoef(valid['temp_c'], valid['usage_pct'])[0, 1]
                stats_dict['temp_usage_corr'] = r
                stats_dict['temp_usage_pval'] = None  # Can't compute p-value without scipy

    # Power stats (if available)
    if available['power']:
        power = df['power_w'].dropna()
        if len(power) > 0:
            stats_dict['power_mean'] = power.mean()
            stats_dict['power_max'] = power.max()
            stats_dict['power_min'] = power.min()

    if available['power_ccd']:
        if 'ccd0_w' in df.columns:
            ccd0 = df['ccd0_w'].dropna()
            if len(ccd0) > 0:
                stats_dict['ccd0_mean'] = ccd0.mean()
        if 'ccd1_w' in df.columns:
            ccd1 = df['ccd1_w'].dropna()
            if len(ccd1) > 0:
                stats_dict['ccd1_mean'] = ccd1.mean()

    return stats_dict


def print_statistics(stats_dict: dict, tjmax: float, use_fahrenheit: bool = False,
                     available: dict | None = None):
    """Print formatted statistics to console."""
    unit = "°F" if use_fahrenheit else "°C"
    # Display thresholds in appropriate unit
    thresh_85 = c_to_f(85) if use_fahrenheit else 85
    thresh_90 = c_to_f(90) if use_fahrenheit else 90

    print("\n" + "=" * 60)
    print("CPU ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\n{'Duration:':<25} {stats_dict['duration_min']:.1f} minutes ({stats_dict['samples']} samples)")

    # Temperature stats (only if available)
    if 'temp_mean' in stats_dict:
        print(f"\n{'--- Temperature Stats ---':^60}")
        print(f"{'  Min:':<25} {stats_dict['temp_min']:.2f}{unit}")
        print(f"{'  Max:':<25} {stats_dict['temp_max']:.2f}{unit}")
        print(f"{'  Mean:':<25} {stats_dict['temp_mean']:.2f}{unit}")
        print(f"{'  Std Dev:':<25} {stats_dict['temp_std']:.2f}{unit}")
        print(f"{'  Median:':<25} {stats_dict['temp_median']:.2f}{unit}")

        print(f"\n{'--- Thermal Headroom ---':^60}")
        print(f"{'  TJMax:':<25} {tjmax:.0f}{unit}")
        print(f"{'  Headroom (TJMax - Max):':<25} {stats_dict['headroom']:.2f}{unit}")
        print(f"  Time above {thresh_85:.0f}{unit}:".ljust(25) + f" {stats_dict['time_above_85']:.1f}%")
        print(f"  Time above {thresh_90:.0f}{unit}:".ljust(25) + f" {stats_dict['time_above_90']:.1f}%")
        print(f"{'  Time above TJMax:':<25} {stats_dict['time_above_tjmax']:.1f}%")

    if 'clock_mean' in stats_dict:
        print(f"\n{'--- Clock Speed Stats ---':^60}")
        print(f"{'  Mean:':<25} {stats_dict['clock_mean']:.0f} MHz")
        print(f"{'  Min:':<25} {stats_dict['clock_min']:.0f} MHz")
        print(f"{'  Max:':<25} {stats_dict['clock_max']:.0f} MHz")
        print(f"{'  Std Dev:':<25} {stats_dict['clock_std']:.1f} MHz")

    if 'usage_mean' in stats_dict:
        print(f"\n{'--- CPU Usage Stats ---':^60}")
        print(f"{'  Mean Usage:':<25} {stats_dict['usage_mean']:.1f}%")
        print(f"{'  Min Usage:':<25} {stats_dict['usage_min']:.1f}%")
        print(f"{'  Max Usage:':<25} {stats_dict['usage_max']:.1f}%")

    if 'load_mean' in stats_dict:
        print(f"{'  Mean Load (1m avg):':<25} {stats_dict['load_mean']:.2f}")
        print(f"{'  Max Load (1m avg):':<25} {stats_dict['load_max']:.2f}")

    if 'temp_usage_corr' in stats_dict:
        print(f"\n{'--- Correlation ---':^60}")
        print(f"{'  Temp vs Usage (r):':<25} {stats_dict['temp_usage_corr']:.3f}")
        if stats_dict['temp_usage_pval'] is not None:
            sig = "significant" if stats_dict['temp_usage_pval'] < 0.05 else "not significant"
            print(f"{'  P-value:':<25} {stats_dict['temp_usage_pval']:.2e} ({sig})")
        else:
            print(f"{'  P-value:':<25} N/A (scipy not installed)")

    if 'power_mean' in stats_dict:
        print(f"\n{'--- Power Stats ---':^60}")
        print(f"{'  Mean Power:':<25} {stats_dict['power_mean']:.1f} W")
        print(f"{'  Min Power:':<25} {stats_dict['power_min']:.1f} W")
        print(f"{'  Max Power:':<25} {stats_dict['power_max']:.1f} W")
        if 'ccd0_mean' in stats_dict:
            print(f"{'  CCD0 Mean:':<25} {stats_dict['ccd0_mean']:.1f} W")
        if 'ccd1_mean' in stats_dict:
            print(f"{'  CCD1 Mean:':<25} {stats_dict['ccd1_mean']:.1f} W")

    print("\n" + "=" * 60)


def get_time_locator(df: pd.DataFrame, target_ticks: int = 8):
    """Return an appropriate tick locator based on data duration."""
    duration_minutes = (df.index[-1] - df.index[0]).total_seconds() / 60

    # Calculate ideal interval to get ~target_ticks
    ideal_interval = duration_minutes / target_ticks

    if ideal_interval < 1:
        # Less than 1 minute intervals - use seconds
        seconds = max(10, int(ideal_interval * 60 / 10) * 10)  # Round to 10s
        return mdates.SecondLocator(interval=seconds)
    elif ideal_interval < 60:
        # Use minutes - pick nice round numbers
        for interval in [1, 2, 5, 10, 15, 20, 30]:
            if interval >= ideal_interval * 0.7:
                return mdates.MinuteLocator(interval=interval)
        return mdates.MinuteLocator(interval=30)
    else:
        # Use hours
        hours = max(1, int(ideal_interval / 60))
        return mdates.HourLocator(interval=hours)


def plot_temperature_timeline(df: pd.DataFrame, resample: str, tjmax: float,
                              stats_dict: dict, output_path: str, use_fahrenheit: bool = False):
    """Create temperature timeline with min/max envelope."""
    plt.figure(figsize=(14, 7))
    unit = "°F" if use_fahrenheit else "°C"

    # Resample for smoothing
    temp_resampled = df['temp_c'].resample(resample)
    temp_mean = temp_resampled.mean()
    temp_min = temp_resampled.min()
    temp_max = temp_resampled.max()

    # Convert to Fahrenheit if requested
    if use_fahrenheit:
        temp_mean = temp_mean.apply(c_to_f)
        temp_min = temp_min.apply(c_to_f)
        temp_max = temp_max.apply(c_to_f)
        tjmax_display = c_to_f(tjmax)
        warning_temp = c_to_f(90)
    else:
        tjmax_display = tjmax
        warning_temp = 90

    # Plot envelope
    plt.fill_between(temp_mean.index, temp_min.values, temp_max.values,
                     alpha=0.3, color='tab:red', label='Min/Max Range')

    # Plot mean line
    plt.plot(temp_mean.index, temp_mean.values,
             color='tab:red', linewidth=2, label=f'Mean Temp ({resample} avg)')

    # Reference lines
    plt.axhline(y=tjmax_display, color='black', linestyle='--', linewidth=2,
                label=f'TJMax ({tjmax_display:.0f}{unit})')
    plt.axhline(y=warning_temp, color='orange', linestyle=':', linewidth=1.5,
                label=f'{warning_temp:.0f}{unit} Warning')

    # Stats annotation box
    stats_text = (f"Max: {stats_dict['temp_max']:.1f}{unit}\n"
                  f"Mean: {stats_dict['temp_mean']:.1f}{unit}\n"
                  f"Headroom: {stats_dict['headroom']:.1f}{unit}")
    plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Formatting
    ax = plt.gca()
    ax.xaxis.set_major_locator(get_time_locator(df))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.margins(x=0)  # Remove horizontal padding
    plt.gcf().autofmt_xdate()

    plt.title('CPU Temperature Over Time')
    plt.xlabel('Time')
    plt.ylabel(f'Temperature ({unit})')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def plot_usage_timeline(df: pd.DataFrame, resample: str, stats_dict: dict, output_path: str):
    """Create CPU usage timeline with load average overlay (fallback when no temp data)."""
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Resample for smoothing
    usage_resampled = df['usage_pct'].resample(resample)
    usage_mean = usage_resampled.mean()
    usage_min = usage_resampled.min()
    usage_max = usage_resampled.max()

    # Plot usage envelope
    ax1.fill_between(usage_mean.index, usage_min.values, usage_max.values,
                     alpha=0.3, color='tab:blue', label='Min/Max Range')

    # Plot mean usage line
    ax1.plot(usage_mean.index, usage_mean.values,
             color='tab:blue', linewidth=2, label=f'CPU Usage ({resample} avg)')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('CPU Usage (%)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 105)  # 0-100% with small margin

    # Load average on secondary axis
    if 'load_avg' in df.columns:
        load_resampled = df['load_avg'].resample(resample).mean()

        ax2 = ax1.twinx()
        ax2.plot(load_resampled.index, load_resampled.values,
                 color='tab:orange', linewidth=2, alpha=0.7, label='Load Avg (1m)')
        ax2.set_ylabel('Load Average', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(loc='lower right')

    # Stats annotation box
    stats_text = f"Mean: {stats_dict.get('usage_mean', 0):.1f}%\n"
    stats_text += f"Max: {stats_dict.get('usage_max', 0):.1f}%"
    if 'load_mean' in stats_dict:
        stats_text += f"\nLoad Avg: {stats_dict['load_mean']:.2f}"
    plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Formatting
    ax1.xaxis.set_major_locator(get_time_locator(df))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.margins(x=0)  # Remove horizontal padding
    ax1.grid(True, linestyle='--', alpha=0.4)

    plt.title('CPU Usage Over Time')
    plt.gcf().autofmt_xdate()
    fig.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def plot_clock_analysis(df: pd.DataFrame, resample: str, output_path: str,
                        use_fahrenheit: bool = False, has_temp: bool = True):
    """Create clock speed analysis with temperature overlay."""
    if 'avg_clock_mhz' not in df.columns:
        print("Skipping clock analysis: no clock data found")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))
    unit = "°F" if use_fahrenheit else "°C"

    # Resample
    clock_resampled = df['avg_clock_mhz'].resample(resample).mean()
    usage_resampled = df['usage_pct'].resample(resample).mean() if 'usage_pct' in df.columns else None

    # Clock speed on primary axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Clock Speed (MHz)', color=color1)
    ax1.plot(clock_resampled.index, clock_resampled.values,
             color=color1, linewidth=2, label='Avg Clock Speed')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.xaxis.set_major_locator(get_time_locator(df))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.margins(x=0)  # Remove horizontal padding
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Temperature on secondary axis (if available)
    if has_temp and 'temp_c' in df.columns:
        temp_resampled = df['temp_c'].resample(resample).mean()
        if use_fahrenheit:
            temp_resampled = temp_resampled.apply(c_to_f)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel(f'Temperature ({unit})', color=color2)
        ax2.plot(temp_resampled.index, temp_resampled.values,
                 color=color2, linewidth=2, alpha=0.7, label='Temperature')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        # Without temperature, show usage instead
        if usage_resampled is not None:
            ax2 = ax1.twinx()
            color2 = 'tab:green'
            ax2.set_ylabel('CPU Usage (%)', color=color2)
            ax2.plot(usage_resampled.index, usage_resampled.values,
                     color=color2, linewidth=2, alpha=0.7, label='CPU Usage')
            ax2.tick_params(axis='y', labelcolor=color2)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax1.legend(loc='lower right')

    # Highlight potential throttling: high CPU usage (>80%) but clocks below peak
    max_clock = clock_resampled.max()
    throttle_threshold = max_clock - 100
    clock_low = clock_resampled < throttle_threshold
    if usage_resampled is not None:
        # Only flag when usage is high but clocks are low (actual throttling indicator)
        high_usage = usage_resampled > 80
        throttle_mask = clock_low & high_usage
    else:
        throttle_mask = clock_low

    if throttle_mask.any():
        ax1.fill_between(clock_resampled.index, clock_resampled.min(), clock_resampled.values,
                         where=throttle_mask, alpha=0.3, color='orange',
                         label='Potential Throttling')

    plt.title('Clock Speed Analysis')

    plt.gcf().autofmt_xdate()
    fig.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def plot_temp_vs_load(df: pd.DataFrame, output_path: str, use_fahrenheit: bool = False):
    """Create temperature vs CPU usage scatter plot."""
    if 'usage_pct' not in df.columns:
        print("Skipping correlation plot: no usage data found")
        return

    plt.figure(figsize=(10, 8))
    unit = "°F" if use_fahrenheit else "°C"

    valid = df[['temp_c', 'usage_pct']].dropna().copy()
    if use_fahrenheit:
        valid['temp_c'] = valid['temp_c'].apply(c_to_f)

    # Color by time (position in dataset)
    colors = np.linspace(0, 1, len(valid))

    scatter = plt.scatter(valid['usage_pct'], valid['temp_c'],
                          c=colors, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Time (start → end)')

    # Trend line (only if there's variance in the data)
    usage_range = valid['usage_pct'].max() - valid['usage_pct'].min()
    if len(valid) > 2 and usage_range > 0:
        try:
            if SCIPY_AVAILABLE:
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                    valid['usage_pct'], valid['temp_c'])
            else:
                # Numpy fallback
                coeffs = np.polyfit(valid['usage_pct'], valid['temp_c'], 1)
                slope, intercept = coeffs
                r_value = np.corrcoef(valid['usage_pct'], valid['temp_c'])[0, 1]

            x_line = np.array([valid['usage_pct'].min(), valid['usage_pct'].max()])
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, 'r--', linewidth=2,
                     label=f'Trend (R² = {r_value**2:.3f})')
        except (ValueError, FloatingPointError):
            pass  # Skip trend line if regression fails

    plt.title('Temperature vs CPU Usage')
    plt.xlabel('CPU Usage (%)')
    plt.ylabel(f'Temperature ({unit})')
    plt.grid(True, linestyle='--', alpha=0.4)
    # Only show legend if trend line was added
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend(loc='upper right')

    # Use small margin so edge points aren't clipped against axis
    ax = plt.gca()
    ax.margins(0.02)

    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def plot_temp_histogram(df: pd.DataFrame, tjmax: float, stats_dict: dict, output_path: str,
                        use_fahrenheit: bool = False):
    """Create temperature distribution histogram."""
    plt.figure(figsize=(10, 7))
    unit = "°F" if use_fahrenheit else "°C"

    temp = df['temp_c'].dropna()

    # Keep original Celsius for threshold calculations
    temp_c = temp.copy()

    # Convert for display if needed
    if use_fahrenheit:
        temp = temp.apply(c_to_f)
        tjmax_display = c_to_f(tjmax)
        thresh_85 = c_to_f(85)
        thresh_90 = c_to_f(90)
    else:
        tjmax_display = tjmax
        thresh_85 = 85
        thresh_90 = 90

    # Histogram
    n, bins, patches = plt.hist(temp, bins=30, edgecolor='black', alpha=0.7, color='tab:red')

    # Color bins by thermal zone
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center >= tjmax_display:
            patch.set_facecolor('darkred')
        elif bin_center >= thresh_90:
            patch.set_facecolor('orangered')
        elif bin_center >= thresh_85:
            patch.set_facecolor('orange')

    # Reference lines
    plt.axvline(x=stats_dict['temp_mean'], color='blue', linestyle='-', linewidth=2,
                label=f"Mean: {stats_dict['temp_mean']:.1f}{unit}")
    plt.axvline(x=tjmax_display, color='black', linestyle='--', linewidth=2,
                label=f'TJMax: {tjmax_display:.0f}{unit}')

    # Zone annotations (calculate from Celsius data)
    total = len(temp_c)
    zone_text = (f"<{thresh_85:.0f}{unit}: {(temp_c < 85).sum() / total * 100:.1f}%\n"
                 f"{thresh_85:.0f}-{thresh_90:.0f}{unit}: {((temp_c >= 85) & (temp_c < 90)).sum() / total * 100:.1f}%\n"
                 f"{thresh_90:.0f}-TJMax: {((temp_c >= 90) & (temp_c < tjmax)).sum() / total * 100:.1f}%\n"
                 f"≥TJMax: {(temp_c >= tjmax).sum() / total * 100:.1f}%")
    plt.annotate(zone_text, xy=(0.02, 0.98), xycoords='axes fraction',
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title('Temperature Distribution')
    plt.xlabel(f'Temperature ({unit})')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4, axis='y')

    # Set x-axis limits to histogram bin range (remove margins)
    ax = plt.gca()
    ax.set_xlim(bins[0], bins[-1])

    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def plot_power_analysis(df: pd.DataFrame, resample: str, output_path: str,
                        use_fahrenheit: bool = False, has_temp: bool = True,
                        has_ccd: bool = True):
    """Create power analysis chart with temperature overlay."""
    if 'power_w' not in df.columns:
        print("Skipping power analysis: no power data found")
        return False

    power = df['power_w'].dropna()
    if len(power) == 0:
        print("Skipping power analysis: no valid power data")
        return False

    fig, ax1 = plt.subplots(figsize=(14, 7))
    unit = "°F" if use_fahrenheit else "°C"

    # Resample
    power_resampled = df['power_w'].resample(resample).mean()

    # Check for CCD data
    if has_ccd and 'ccd0_w' in df.columns and 'ccd1_w' in df.columns:
        ccd0_resampled = df['ccd0_w'].resample(resample).mean()
        ccd1_resampled = df['ccd1_w'].resample(resample).mean()
    else:
        has_ccd = False

    # Power on primary axis
    color1 = 'tab:green'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Power (W)', color=color1)
    ax1.plot(power_resampled.index, power_resampled.values,
             color=color1, linewidth=2, label='Socket Power')

    if has_ccd:
        ax1.plot(ccd0_resampled.index, ccd0_resampled.values,
                 color='tab:blue', linewidth=1.5, alpha=0.7, label='CCD0 Power')
        ax1.plot(ccd1_resampled.index, ccd1_resampled.values,
                 color='tab:purple', linewidth=1.5, alpha=0.7, label='CCD1 Power')

    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.xaxis.set_major_locator(get_time_locator(df))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.margins(x=0)  # Remove horizontal padding
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Temperature on secondary axis (if available)
    if has_temp and 'temp_c' in df.columns:
        temp_resampled = df['temp_c'].resample(resample).mean()
        if use_fahrenheit:
            temp_resampled = temp_resampled.apply(c_to_f)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel(f'Temperature ({unit})', color=color2)
        ax2.plot(temp_resampled.index, temp_resampled.values,
                 color=color2, linewidth=2, alpha=0.7, label='Temperature')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(loc='lower right')

    plt.title('Power Consumption Analysis')

    plt.gcf().autofmt_xdate()
    fig.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    return True


def main():
    args = parse_args()
    use_f = args.fahrenheit

    # Handle output path - if it's a directory, append default base name
    output_base = args.output
    output_path = Path(output_base)
    if output_path.is_dir() or output_base.endswith('/'):
        output_path.mkdir(parents=True, exist_ok=True)
        output_base = str(output_path / 'cpu_analysis')

    df = load_data(args.input)
    print(f"Loaded {len(df)} samples from: {args.input}")

    # Detect available data
    available = detect_available_data(df)

    # Print data availability
    print("\nData availability:")
    print(f"  Temperature: {'Yes' if available['temperature'] else 'No'}")
    print(f"  CPU Usage:   {'Yes' if available['usage'] else 'No'}")
    print(f"  Power:       {'Yes' if available['power'] else 'No'}")
    print(f"  Power CCD:   {'Yes' if available['power_ccd'] else 'No'}")
    print(f"  Clocks:      {'Yes' if available['clocks'] else 'No'}")

    if not SCIPY_AVAILABLE:
        print("\nNote: scipy not installed, using numpy for correlation (no p-values)")

    # Convert tjmax if using Fahrenheit
    tjmax_display = c_to_f(args.tjmax) if use_f else args.tjmax

    # Compute and print statistics
    stats_dict = compute_statistics(df, args.tjmax, use_f, available)
    print_statistics(stats_dict, tjmax_display, use_f, available)

    # Generate plots based on available data
    print("\nGenerating visualizations...")

    output_files = []

    # Temperature timeline (requires temperature)
    if available['temperature']:
        output_path = f"{output_base}_temp.png"
        plot_temperature_timeline(df, args.resample, args.tjmax, stats_dict, output_path, use_f)
        output_files.append(output_path)
    else:
        print("Skipping temperature timeline: no temperature data")

    # Usage timeline (fallback when no temperature, or always available)
    if available['usage']:
        output_path = f"{output_base}_usage.png"
        plot_usage_timeline(df, args.resample, stats_dict, output_path)
        output_files.append(output_path)

    # Clock analysis (requires clocks)
    if available['clocks']:
        output_path = f"{output_base}_clocks.png"
        plot_clock_analysis(df, args.resample, output_path, use_f, available['temperature'])
        output_files.append(output_path)
    else:
        print("Skipping clock analysis: no clock data")

    # Correlation plot (requires temperature + usage)
    if available['temperature'] and available['usage']:
        output_path = f"{output_base}_correlation.png"
        plot_temp_vs_load(df, output_path, use_f)
        output_files.append(output_path)
    else:
        print("Skipping correlation plot: requires both temperature and usage data")

    # Histogram (requires temperature)
    if available['temperature']:
        output_path = f"{output_base}_histogram.png"
        plot_temp_histogram(df, args.tjmax, stats_dict, output_path, use_f)
        output_files.append(output_path)
    else:
        print("Skipping temperature histogram: no temperature data")

    # Power analysis (requires power)
    if available['power']:
        output_path = f"{output_base}_power.png"
        if plot_power_analysis(df, args.resample, output_path, use_f,
                                available['temperature'], available['power_ccd']):
            output_files.append(output_path)

    if not output_files:
        print("\nWarning: No graphs could be generated - insufficient data")
        sys.exit(1)

    print(f"\nAnalysis complete! Generated {len(output_files)} graph(s).")

    # Open images with default viewer
    if args.open:
        opener = None
        if shutil.which('xdg-open'):
            opener = 'xdg-open'
        elif shutil.which('open'):  # macOS
            opener = 'open'
        elif shutil.which('start'):  # Windows
            opener = 'start'

        if opener:
            for f in output_files:
                if Path(f).exists():
                    subprocess.Popen(
                        [opener, f],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True
                    )
        else:
            print("Warning: Could not find a command to open images")

    if not args.no_show:
        # Only call show() if using an interactive backend
        backend = plt.get_backend()
        if 'inline' in backend.lower() or 'agg' in backend.lower():
            pass  # Non-interactive, skip show()
        else:
            plt.show()


if __name__ == '__main__':
    main()
