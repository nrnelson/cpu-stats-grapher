# CPU Stats Grapher

A CPU thermal analysis toolkit for monitoring and visualizing CPU temperature, clock speeds, power consumption, and usage metrics. Supports AMD Ryzen/EPYC, Intel Core processors, and falls back gracefully on systems with limited hardware monitoring.

## Features

- **Real-time data collection** via shell script (`cpu-stats-logger.sh`)
- **Comprehensive visualization** via Python script (`cpu-stats-grapher.py`)
- Temperature timeline with min/max envelope
- Clock speed analysis with throttling detection
- Power consumption tracking (socket + per-CCD breakdown)
- Temperature vs CPU usage correlation with trend lines
- Temperature distribution histogram with thermal zone coloring
- CPU usage timeline (fallback when no temperature data)
- Statistical summary (min, max, mean, std dev, percentiles)

## Graceful Degradation

Both scripts automatically detect available hardware monitoring capabilities and adapt accordingly:

### Supported Temperature Sources (in priority order)

| Priority | Source | Systems |
|----------|--------|---------|
| 1 | k10temp | AMD Ryzen/EPYC (via lm-sensors) |
| 2 | coretemp | Intel Core (via lm-sensors) |
| 3 | acpitz | ACPI thermal (laptops, less accurate) |
| 4 | thermal_zone | Kernel fallback (all Linux, least accurate) |

### Supported Power Sources (in priority order)

| Priority | Source | Systems | Per-Core |
|----------|--------|---------|----------|
| 1 | zenergy | AMD (kernel module) | Yes (CCD breakdown) |
| 2 | amd_energy | AMD (official driver) | Yes (CCD breakdown) |
| 3 | powercap RAPL | Intel & AMD | No (package only) |

### Supported Frequency Sources

| Priority | Source | Notes |
|----------|--------|-------|
| 1 | cpufreq | Scaling driver (most accurate) |
| 2 | /proc/cpuinfo | Fallback (less accurate) |

### Minimum Requirements

The logger requires at least ONE of:
- Temperature monitoring
- Power monitoring
- Clock speed monitoring

If none are available (only basic /proc data), the logger exits with an error rather than collecting useless data.

## Requirements

### System
- Linux with hardware monitoring support
- `lm-sensors` package (recommended but optional)
- One of the supported CPU drivers loaded

### Python
- Python 3.8+
- Required: `matplotlib`, `numpy`, `pandas`
- Optional: `scipy` (for p-values in correlation analysis)

## Installation

```bash
git clone https://github.com/nrnelson/cpu-stats-grapher.git
cd cpu-stats-grapher
pip install -r requirements.txt
```

## Usage

### 1. Collect Data

Run the logger script to collect CPU metrics:

```bash
./cpu-stats-logger.sh > cpu_temps.log
```

Press `Ctrl+C` to stop collection. The script samples at 1 Hz with anti-drift timing.

### Logger Command-Line Options

```
./cpu-stats-logger.sh [OPTIONS]

Options:
  -t, --temp         Collect temperature data only
  -p, --power        Collect power data only
  -c, --clocks       Collect clock speed data only
  -a, --all          Collect all available metrics (default)
  -i, --interval N   Polling interval in seconds (default: 1)
  -h, --help         Show help message

Combined flags:
  -tc             Temperature + clocks
  -tp             Temperature + power
  -tpc            All metrics (same as -a)
```

CPU usage and load average are always included for context.

**Examples:**

```bash
# Collect all available metrics at 1 Hz (default)
./cpu-stats-logger.sh > cpu_temps.log

# Collect every 5 seconds (lower overhead for long runs)
./cpu-stats-logger.sh -i 5 > cpu_temps.log

# Temperature only
./cpu-stats-logger.sh -t > temps_only.log

# Temperature + clocks (no power)
./cpu-stats-logger.sh -tc > temps_clocks.log

# See what's detected without collecting
./cpu-stats-logger.sh -h
```

### 2. Analyze Data

Run the grapher to generate visualizations:

```bash
python cpu-stats-grapher.py cpu_temps.log
```

This generates PNG files and prints a statistical summary to the console.

### Grapher Command-Line Options

```
python cpu-stats-grapher.py [OPTIONS] INPUT_FILE

Options:
  -o, --output NAME    Base name for output files (default: cpu_analysis)
  --resample INTERVAL  Resample interval for smoothing (default: 10s)
  --tjmax TEMP         Thermal junction max temperature in Celsius (default: 95)
  --no-show            Skip displaying plots (for headless use)
  --open               Open generated images with default viewer
  -f, --fahrenheit     Display temperatures in Fahrenheit
```

### Examples

```bash
# Basic analysis
python cpu-stats-grapher.py cpu_temps.log

# Custom output name and 5-second smoothing
python cpu-stats-grapher.py cpu_temps.log -o my_analysis --resample 5s

# Set custom TJMax and skip interactive display
python cpu-stats-grapher.py cpu_temps.log --tjmax 90 --no-show

# Display temperatures in Fahrenheit
python cpu-stats-grapher.py cpu_temps.log -f
```

## Output Files

The grapher produces these PNG files based on available data:

| File | Requirements | Description |
|------|--------------|-------------|
| `*_temp.png` | Temperature | Temperature timeline with min/max envelope |
| `*_usage.png` | Usage | CPU usage timeline with load average |
| `*_clocks.png` | Clocks | Clock speed analysis with temp/usage overlay |
| `*_correlation.png` | Temp + Usage | Temperature vs CPU usage scatter plot |
| `*_histogram.png` | Temperature | Temperature distribution with thermal zones |
| `*_power.png` | Power | Power consumption analysis |

If data is missing for a graph, it's skipped with a message.

## Startup Banner

When the logger starts, it displays detected components:

```
=== CPU Stats Logger - Component Detection ===
[OK]   Temperature: k10temp via lm-sensors (k10temp-pci-00c3)
[OK]   CPU Usage: /proc/stat
[OK]   Power: zenergy (AMD per-core)
[OK]   Clock Speeds: cpufreq scaling driver
================================================
```

Status indicators:
- `[OK]` - Component detected and will be used
- `[SKIP]` - Component requested but not available
- `[----]` - Component not requested

## Troubleshooting

### No temperature source detected

1. Install lm-sensors: `sudo apt install lm-sensors`
2. Detect sensors: `sudo sensors-detect`
3. Load the appropriate kernel module (k10temp for AMD, coretemp for Intel)

### No power source detected

**AMD systems:**
- Install zenergy: Check your distribution's packages or build from [source](https://github.com/bouletmarc/HWiNFO-Shared-Memory-Dump)
- Or use amd_energy: `sudo modprobe amd_energy`

**Intel systems:**
- Enable RAPL: Usually available by default at `/sys/class/powercap/intel-rapl/`

### Requested metric unavailable

If you explicitly request a metric that isn't available, the logger exits with an error:

```bash
$ ./cpu-stats-logger.sh --power
[ERROR] Power requested but no source available
```

### scipy not installed

The grapher works without scipy but won't compute p-values for correlations:

```
Note: scipy not installed, using numpy for correlation (no p-values)
```

Install scipy for full functionality: `pip install scipy`

## Data Format

The logger outputs tab-separated values. The columns vary based on available/requested metrics:

**Always included:**
```
timestamp  load_avg  usage_pct
```

**With temperature (`-t`):**
```
timestamp  load_avg  usage_pct  temp_c
```

**With power (`-p`, AMD with CCD):**
```
timestamp  load_avg  usage_pct  power_w  ccd0_w  ccd1_w
```

**With power (`-p`, Intel/RAPL):**
```
timestamp  load_avg  usage_pct  power_w
```

**With clocks (`-c`):**
```
timestamp  load_avg  usage_pct  core0_mhz  core1_mhz  ...
```

The grapher auto-detects the format and generates appropriate visualizations.

## License

MIT License - see [LICENSE](LICENSE) for details.
