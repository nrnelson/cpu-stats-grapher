# CPU Stats Grapher

A CPU thermal analysis toolkit for monitoring and visualizing CPU temperature, clock speeds, power consumption, and usage metrics. Designed for AMD Ryzen processors with zenergy support.

## Features

- **Real-time data collection** via shell script (`cpu-stats-logger.sh`)
- **Comprehensive visualization** via Python script (`cpu-stats-grapher.py`)
- Temperature timeline with min/max envelope
- Clock speed analysis with throttling detection
- Power consumption tracking (socket + per-CCD breakdown)
- Temperature vs CPU usage correlation with trend lines
- Temperature distribution histogram with thermal zone coloring
- Statistical summary (min, max, mean, std dev, percentiles)

## Requirements

### System
- Linux with `lm-sensors` installed
- AMD CPU with `k10temp` sensor support
- `zenergy` kernel module for power monitoring (optional but recommended)

### Python
- Python 3.8+
- Dependencies listed in `requirements.txt`

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

### 2. Analyze Data

Run the grapher to generate visualizations:

```bash
python cpu-stats-grapher.py cpu_temps.log
```

This generates PNG files and prints a statistical summary to the console.

### Command-Line Options

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

The grapher produces these PNG files:

| File | Description |
|------|-------------|
| `*_temp.png` | Temperature timeline with min/max envelope |
| `*_clocks.png` | Clock speed analysis with temperature overlay |
| `*_correlation.png` | Temperature vs CPU usage scatter plot |
| `*_histogram.png` | Temperature distribution with thermal zones |
| `*_power.png` | Power consumption analysis (if power data available) |

## Hardware Configuration

### Sensor Configuration

The logger script is configured for AMD Ryzen CPUs. Edit these variables in `cpu-stats-logger.sh` if needed:

```bash
SENSOR="k10temp-pci-00c3"  # Your CPU temp sensor name
INPUT="temp1_input"         # Temperature input to read
TEMP_LIMIT=89.0            # Throttling detection threshold
```

To find your sensor name:
```bash
sensors
```

### Power Monitoring

Power monitoring requires the `zenergy` kernel module. Install it from your distribution's packages or build from source. The script auto-detects the hwmon path.

## Data Format

The logger outputs tab-separated values:

```
timestamp  temp_c  load_avg  usage_pct  throttled  power_w  ccd0_w  ccd1_w  core0_mhz...
```

The grapher handles both old format (without power columns) and new format (with power data).

## License

MIT License - see [LICENSE](LICENSE) for details.
