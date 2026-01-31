#!/bin/bash
#
# CPU Stats Logger - Real-time CPU metrics collector
#
# Author: Nathan Nelson <nrnelson@gmail.com>
# Source: https://github.com/nrnelson/cpu-stats-grapher
#
# Collects temperature, clock speed, power consumption, and usage data
# at 1 Hz with anti-drift timing for accurate sampling intervals.
#
# Usage:
#   ./cpu-stats-logger.sh > output.log
#   ./cpu-stats-logger.sh -t          # Temperature only
#   ./cpu-stats-logger.sh -tc         # Temperature + clocks
#   ./cpu-stats-logger.sh | tee output.log
#
# Supports: AMD (k10temp, zenergy), Intel (coretemp, RAPL), thermal_zone fallback
#
# Output: Tab-separated values with header row
# Stop with Ctrl+C
#

set -o pipefail

# --- CONFIGURATION ---
# (No user-configurable options currently)
# ---------------------

# --- LOGGING FUNCTIONS ---
log_info() {
    echo "[INFO]  $*" >&2
}

log_warn() {
    echo "[WARN]  $*" >&2
}

log_error() {
    echo "[ERROR] $*" >&2
}

# --- USAGE ---
print_usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") [OPTIONS]

Collect CPU metrics (temperature, power, clock speeds) at a configurable interval.

Options:
  -t, --temp         Collect temperature data
  -p, --power        Collect power data
  -c, --clocks       Collect clock speed data
  -a, --all          Collect all available metrics (default)
  -i, --interval N   Polling interval in seconds (default: 1)
  -h, --help         Show this help message

Examples:
  $(basename "$0")              # All available metrics at 1 Hz
  $(basename "$0") -i 5         # All metrics every 5 seconds
  $(basename "$0") -t           # Temperature only
  $(basename "$0") -tc          # Temperature + clocks
  $(basename "$0") --power      # Power only

CPU usage and load average are always included for context.
EOF
}

# --- CLI ARGUMENT PARSING ---
COLLECT_TEMP=false
COLLECT_POWER=false
COLLECT_CLOCKS=false
EXPLICIT_REQUEST=false
POLL_INTERVAL=1

parse_args() {
    # If no args, default to all
    if [[ $# -eq 0 ]]; then
        COLLECT_TEMP=true
        COLLECT_POWER=true
        COLLECT_CLOCKS=true
        return
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -t|--temp)
                COLLECT_TEMP=true
                EXPLICIT_REQUEST=true
                shift
                ;;
            -p|--power)
                COLLECT_POWER=true
                EXPLICIT_REQUEST=true
                shift
                ;;
            -c|--clocks)
                COLLECT_CLOCKS=true
                EXPLICIT_REQUEST=true
                shift
                ;;
            -a|--all)
                COLLECT_TEMP=true
                COLLECT_POWER=true
                COLLECT_CLOCKS=true
                shift
                ;;
            -i|--interval)
                if [[ -z "$2" || "$2" == -* ]]; then
                    log_error "Option $1 requires an argument"
                    exit 2
                fi
                if ! [[ "$2" =~ ^[0-9]+$ ]] || [[ "$2" -lt 1 ]]; then
                    log_error "Interval must be a positive integer (got: $2)"
                    exit 2
                fi
                POLL_INTERVAL="$2"
                shift 2
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            -*)
                # Handle combined short flags like -tc, -tpc, etc.
                flags="${1#-}"
                shift
                for ((i=0; i<${#flags}; i++)); do
                    flag="${flags:$i:1}"
                    case "$flag" in
                        t) COLLECT_TEMP=true; EXPLICIT_REQUEST=true ;;
                        p) COLLECT_POWER=true; EXPLICIT_REQUEST=true ;;
                        c) COLLECT_CLOCKS=true; EXPLICIT_REQUEST=true ;;
                        a) COLLECT_TEMP=true; COLLECT_POWER=true; COLLECT_CLOCKS=true ;;
                        *)
                            log_error "Unknown flag: -$flag"
                            print_usage
                            exit 2
                            ;;
                    esac
                done
                ;;
            *)
                log_error "Unknown argument: $1"
                print_usage
                exit 2
                ;;
        esac
    done
}

# --- COMPONENT DETECTION ---

# Temperature source detection
# Priority: k10temp > coretemp > acpitz > thermal_zone
# We prefer direct hwmon reads over lm-sensors for speed
TEMP_SOURCE=""
TEMP_SENSOR=""
TEMP_HWMON_PATH=""  # Direct hwmon path (faster than sensors command)
THERMAL_ZONE_PATH=""

detect_temperature_source() {
    # Try direct hwmon reads first (much faster than sensors command)
    # k10temp (AMD)
    local hwmon_path
    hwmon_path=$(grep -l k10temp /sys/class/hwmon/hwmon*/name 2>/dev/null | head -1)
    if [[ -n "$hwmon_path" ]]; then
        local hwmon_dir
        hwmon_dir=$(dirname "$hwmon_path")
        if [[ -r "$hwmon_dir/temp1_input" ]]; then
            TEMP_SOURCE="k10temp"
            TEMP_SENSOR="k10temp (hwmon direct)"
            TEMP_HWMON_PATH="$hwmon_dir/temp1_input"
            return 0
        fi
    fi

    # coretemp (Intel)
    hwmon_path=$(grep -l coretemp /sys/class/hwmon/hwmon*/name 2>/dev/null | head -1)
    if [[ -n "$hwmon_path" ]]; then
        local hwmon_dir
        hwmon_dir=$(dirname "$hwmon_path")
        if [[ -r "$hwmon_dir/temp1_input" ]]; then
            TEMP_SOURCE="coretemp"
            TEMP_SENSOR="coretemp (hwmon direct)"
            TEMP_HWMON_PATH="$hwmon_dir/temp1_input"
            return 0
        fi
    fi

    # acpitz (ACPI thermal)
    hwmon_path=$(grep -l acpitz /sys/class/hwmon/hwmon*/name 2>/dev/null | head -1)
    if [[ -n "$hwmon_path" ]]; then
        local hwmon_dir
        hwmon_dir=$(dirname "$hwmon_path")
        if [[ -r "$hwmon_dir/temp1_input" ]]; then
            TEMP_SOURCE="acpitz"
            TEMP_SENSOR="acpitz (hwmon direct)"
            TEMP_HWMON_PATH="$hwmon_dir/temp1_input"
            return 0
        fi
    fi

    # Fallback to thermal_zone (all Linux systems)
    for zone in /sys/class/thermal/thermal_zone*/temp; do
        if [[ -r "$zone" ]]; then
            TEMP_SOURCE="thermal_zone"
            THERMAL_ZONE_PATH="$zone"
            return 0
        fi
    done

    # No temperature source found
    TEMP_SOURCE=""
    return 1
}

# Power source detection
# Priority: zenergy > amd_energy > powercap RAPL
POWER_SOURCE=""
ZENERGY_HWMON=""
RAPL_PATH=""
POWER_HAS_CCD=false

detect_power_source() {
    # Try zenergy (AMD)
    local zenergy_path
    zenergy_path=$(grep -l "zenergy" /sys/class/hwmon/hwmon*/name 2>/dev/null | head -1)
    if [[ -n "$zenergy_path" ]]; then
        POWER_SOURCE="zenergy"
        ZENERGY_HWMON=$(dirname "$zenergy_path")
        POWER_HAS_CCD=true
        return 0
    fi

    # Try amd_energy (official AMD driver)
    local amd_energy_path
    amd_energy_path=$(grep -l "amd_energy" /sys/class/hwmon/hwmon*/name 2>/dev/null | head -1)
    if [[ -n "$amd_energy_path" ]]; then
        POWER_SOURCE="amd_energy"
        ZENERGY_HWMON=$(dirname "$amd_energy_path")
        POWER_HAS_CCD=true
        return 0
    fi

    # Try powercap RAPL (works for both Intel and AMD)
    if [[ -d /sys/class/powercap/intel-rapl ]]; then
        # Find the package energy file
        for rapl_dir in /sys/class/powercap/intel-rapl/intel-rapl:*/; do
            if [[ -r "${rapl_dir}energy_uj" ]]; then
                POWER_SOURCE="rapl"
                RAPL_PATH="${rapl_dir}energy_uj"
                POWER_HAS_CCD=false
                return 0
            fi
        done
    fi

    # No power source found
    POWER_SOURCE=""
    return 1
}

# Frequency source detection
# Priority: cpufreq > /proc/cpuinfo
FREQ_SOURCE=""

detect_frequency_source() {
    # Try cpufreq scaling driver
    if [[ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq ]]; then
        FREQ_SOURCE="cpufreq"
        return 0
    fi

    # Fallback to /proc/cpuinfo
    if grep -q "cpu MHz" /proc/cpuinfo 2>/dev/null; then
        FREQ_SOURCE="procfs"
        return 0
    fi

    # No frequency source
    FREQ_SOURCE=""
    return 1
}

# --- STARTUP BANNER ---
print_banner() {
    echo "=== CPU Stats Logger - Component Detection ===" >&2

    # Temperature
    if [[ -n "$TEMP_SOURCE" ]]; then
        case "$TEMP_SOURCE" in
            k10temp)   echo "[OK]   Temperature: k10temp via lm-sensors ($TEMP_SENSOR)" >&2 ;;
            coretemp)  echo "[OK]   Temperature: coretemp via lm-sensors ($TEMP_SENSOR)" >&2 ;;
            acpitz)    echo "[OK]   Temperature: acpitz via lm-sensors (ACPI, less accurate)" >&2 ;;
            thermal_zone) echo "[OK]   Temperature: thermal_zone (kernel fallback, less accurate)" >&2 ;;
        esac
    else
        if $COLLECT_TEMP; then
            echo "[SKIP] Temperature: no source available" >&2
        else
            echo "[----] Temperature: not requested" >&2
        fi
    fi

    # Always show CPU usage
    echo "[OK]   CPU Usage: /proc/stat" >&2

    # Power
    if [[ -n "$POWER_SOURCE" ]]; then
        case "$POWER_SOURCE" in
            zenergy)    echo "[OK]   Power: zenergy (AMD per-core)" >&2 ;;
            amd_energy) echo "[OK]   Power: amd_energy (AMD per-core)" >&2 ;;
            rapl)       echo "[OK]   Power: powercap RAPL (package-level only)" >&2 ;;
        esac
    else
        if $COLLECT_POWER; then
            echo "[SKIP] Power: no source available" >&2
        else
            echo "[----] Power: not requested" >&2
        fi
    fi

    # Clocks
    if [[ -n "$FREQ_SOURCE" ]]; then
        case "$FREQ_SOURCE" in
            cpufreq) echo "[OK]   Clock Speeds: cpufreq scaling driver" >&2 ;;
            procfs)  echo "[OK]   Clock Speeds: /proc/cpuinfo (less accurate)" >&2 ;;
        esac
    else
        if $COLLECT_CLOCKS; then
            echo "[SKIP] Clock Speeds: no source available" >&2
        else
            echo "[----] Clock Speeds: not requested" >&2
        fi
    fi

    echo "================================================" >&2
}

# --- DATA COLLECTION FUNCTIONS ---

get_temperature() {
    if [[ -z "$TEMP_SOURCE" ]]; then
        echo "-"
        return
    fi

    local millideg whole frac
    case "$TEMP_SOURCE" in
        k10temp|coretemp|acpitz)
            # Direct hwmon read (millidegrees C)
            millideg=$(< "$TEMP_HWMON_PATH")
            ;;
        thermal_zone)
            # thermal_zone also gives millidegrees
            millideg=$(< "$THERMAL_ZONE_PATH")
            ;;
        *)
            echo "-"
            return
            ;;
    esac

    if [[ -n "$millideg" && "$millideg" =~ ^[0-9]+$ ]]; then
        # Pure bash: convert millidegrees to degrees with 1 decimal
        # e.g., 45250 -> 45.2
        whole=$((millideg / 1000))
        frac=$(( (millideg % 1000) / 100 ))
        echo "${whole}.${frac}"
    else
        echo "-"
    fi
}

# Initialize power tracking variables
PREV_SOCKET_ENERGY=0
PREV_CCD0_ENERGY=0
PREV_CCD1_ENERGY=0
PREV_RAPL_ENERGY=0

get_frequencies() {
    if [[ -z "$FREQ_SOURCE" ]]; then
        return
    fi

    case "$FREQ_SOURCE" in
        cpufreq)
            cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | awk '{printf "%.0f\t", $1/1000}'
            ;;
        procfs)
            awk '/cpu MHz/ {printf "%.0f\t", $4}' /proc/cpuinfo
            ;;
    esac
}

# --- MAIN ---

parse_args "$@"

# Detect available components
detect_temperature_source
detect_power_source
detect_frequency_source

# Disable collection for unavailable sources (unless explicitly requested)
TEMP_AVAILABLE=$([[ -n "$TEMP_SOURCE" ]] && echo true || echo false)
POWER_AVAILABLE=$([[ -n "$POWER_SOURCE" ]] && echo true || echo false)
FREQ_AVAILABLE=$([[ -n "$FREQ_SOURCE" ]] && echo true || echo false)

# Check if explicitly requested metrics are available
if $EXPLICIT_REQUEST; then
    if $COLLECT_TEMP && ! $TEMP_AVAILABLE; then
        log_error "Temperature requested but no source available"
        exit 2
    fi
    if $COLLECT_POWER && ! $POWER_AVAILABLE; then
        log_error "Power requested but no source available"
        exit 2
    fi
    if $COLLECT_CLOCKS && ! $FREQ_AVAILABLE; then
        log_error "Clock speeds requested but no source available"
        exit 2
    fi
fi

# Adjust collection flags based on availability
$COLLECT_TEMP && ! $TEMP_AVAILABLE && COLLECT_TEMP=false
$COLLECT_POWER && ! $POWER_AVAILABLE && COLLECT_POWER=false
$COLLECT_CLOCKS && ! $FREQ_AVAILABLE && COLLECT_CLOCKS=false

# Check minimum viable data requirement
if ! $TEMP_AVAILABLE && ! $POWER_AVAILABLE && ! $FREQ_AVAILABLE; then
    log_error "No hardware metrics available (temperature, power, or clock speeds)"
    log_error "Only basic /proc data would be collected - exiting"
    exit 2
fi

# Final check: at least one metric must be collected
if ! $COLLECT_TEMP && ! $COLLECT_POWER && ! $COLLECT_CLOCKS; then
    log_error "No metrics to collect"
    exit 2
fi

# Print banner
print_banner

# Initialize CPU usage tracking
PREV_TOTAL=0
PREV_IDLE=0

# Build dynamic header
CORE_COUNT=$(nproc)
HEADER="timestamp\tload_avg\tusage_pct"

if $COLLECT_TEMP; then
    HEADER+="\ttemp_c"
fi

if $COLLECT_POWER; then
    if $POWER_HAS_CCD; then
        HEADER+="\tpower_w\tccd0_w\tccd1_w"
    else
        HEADER+="\tpower_w"
    fi
fi

if $COLLECT_CLOCKS; then
    for ((i=0; i<CORE_COUNT; i++)); do
        HEADER+="\tcore${i}_mhz"
    done
fi

echo -e "$HEADER"

# If collecting power, take initial baseline reading and wait 1 second
# so the first output line has valid power data (not "-")
# We only need 1 second regardless of POLL_INTERVAL to establish a delta
if $COLLECT_POWER; then
    case "$POWER_SOURCE" in
        zenergy|amd_energy)
            PREV_SOCKET_ENERGY=$(cat "$ZENERGY_HWMON/energy17_input" 2>/dev/null || echo 0)
            PREV_CCD0_ENERGY=0
            PREV_CCD1_ENERGY=0
            for i in {1..8}; do
                val=$(cat "$ZENERGY_HWMON/energy${i}_input" 2>/dev/null || echo 0)
                PREV_CCD0_ENERGY=$((PREV_CCD0_ENERGY + val))
            done
            for i in {9..16}; do
                val=$(cat "$ZENERGY_HWMON/energy${i}_input" 2>/dev/null || echo 0)
                PREV_CCD1_ENERGY=$((PREV_CCD1_ENERGY + val))
            done
            ;;
        rapl)
            PREV_RAPL_ENERGY=$(cat "$RAPL_PATH" 2>/dev/null || echo 0)
            ;;
    esac
    # Also take initial CPU usage reading for accurate first sample
    read -r cpu user nice system idle iowait irq softirq steal guest rest < /proc/stat
    PREV_TOTAL=$((user + nice + system + idle + iowait + irq + softirq + steal))
    PREV_IDLE="$idle"
    # Wait 1 second to establish baseline (regardless of poll interval)
    sleep 1
    # First iteration uses 1-second delta, subsequent use POLL_INTERVAL
    POWER_DIVISOR=1
    FIRST_POWER_READING=true
fi

# Main collection loop
while true; do
    # 1. Get Time (using bash builtin - much faster than date command)
    printf -v DATE_TIME '%(%Y-%m-%d\t%H:%M:%S)T' -1

    # 2. Get Load Average (1-minute avg only, using read builtin - faster than awk)
    read -r LOAD_AVG _ < /proc/loadavg

    # 3. Calculate CPU Usage %
    read -r cpu user nice system idle iowait irq softirq steal guest rest < /proc/stat

    TOTAL=$((user + nice + system + idle + iowait + irq + softirq + steal))

    DIFF_IDLE=$((idle - PREV_IDLE))
    DIFF_TOTAL=$((TOTAL - PREV_TOTAL))

    if [[ "$DIFF_TOTAL" -ne 0 ]]; then
        CPU_USAGE=$(( (100 * (DIFF_TOTAL - DIFF_IDLE)) / DIFF_TOTAL ))
    else
        CPU_USAGE=0
    fi

    PREV_TOTAL="$TOTAL"
    PREV_IDLE="$idle"

    # Build output line
    OUTPUT="${DATE_TIME}\t${LOAD_AVG}\t${CPU_USAGE}%"

    # 4. Temperature (if collecting)
    if $COLLECT_TEMP; then
        TEMP_C=$(get_temperature)
        OUTPUT+="\t${TEMP_C}"
    fi

    # 5. Power (if collecting)
    # Note: Power collection is inlined here (not in a function) because we need
    # to track previous energy values across iterations, and subshells lose state
    if $COLLECT_POWER; then
        case "$POWER_SOURCE" in
            zenergy|amd_energy)
                # Read all energy values efficiently with single awk calls
                # Socket total is energy17, CCD0 is sum of 1-8, CCD1 is sum of 9-16
                CURR_SOCKET_ENERGY=$(cat "$ZENERGY_HWMON/energy17_input" 2>/dev/null || echo 0)
                CCD0_ENERGY=$(awk '{s+=$1} END{print s}' "$ZENERGY_HWMON"/energy{1,2,3,4,5,6,7,8}_input 2>/dev/null || echo 0)
                CCD1_ENERGY=$(awk '{s+=$1} END{print s}' "$ZENERGY_HWMON"/energy{9,10,11,12,13,14,15,16}_input 2>/dev/null || echo 0)

                if [[ "$PREV_SOCKET_ENERGY" -ne 0 ]]; then
                    # Delta in microjoules, convert to Watts (divide by time interval and 1000000)
                    DELTA_SOCKET=$((CURR_SOCKET_ENERGY - PREV_SOCKET_ENERGY))
                    DELTA_CCD0=$((CCD0_ENERGY - PREV_CCD0_ENERGY))
                    DELTA_CCD1=$((CCD1_ENERGY - PREV_CCD1_ENERGY))

                    # Handle counter wraparound
                    [[ "$DELTA_SOCKET" -lt 0 ]] && DELTA_SOCKET=0
                    [[ "$DELTA_CCD0" -lt 0 ]] && DELTA_CCD0=0
                    [[ "$DELTA_CCD1" -lt 0 ]] && DELTA_CCD1=0

                    # Calculate all three power values in single awk call
                    read -r SOCKET_POWER CCD0_POWER CCD1_POWER <<< $(awk "BEGIN {
                        printf \"%.1f %.1f %.1f\", $DELTA_SOCKET/1000000/$POWER_DIVISOR, $DELTA_CCD0/1000000/$POWER_DIVISOR, $DELTA_CCD1/1000000/$POWER_DIVISOR
                    }")
                    OUTPUT+="\t${SOCKET_POWER}\t${CCD0_POWER}\t${CCD1_POWER}"

                    # After first reading, switch to using POLL_INTERVAL
                    if $FIRST_POWER_READING; then
                        POWER_DIVISOR="$POLL_INTERVAL"
                        FIRST_POWER_READING=false
                    fi
                else
                    OUTPUT+="\t-\t-\t-"
                fi

                PREV_SOCKET_ENERGY="$CURR_SOCKET_ENERGY"
                PREV_CCD0_ENERGY="$CCD0_ENERGY"
                PREV_CCD1_ENERGY="$CCD1_ENERGY"
                ;;
            rapl)
                CURR_RAPL_ENERGY=$(cat "$RAPL_PATH" 2>/dev/null || echo 0)

                if [[ "$PREV_RAPL_ENERGY" -ne 0 ]]; then
                    DELTA=$((CURR_RAPL_ENERGY - PREV_RAPL_ENERGY))
                    [[ "$DELTA" -lt 0 ]] && DELTA=0
                    POWER=$(awk "BEGIN {printf \"%.1f\", $DELTA / 1000000 / $POWER_DIVISOR}")
                    OUTPUT+="\t${POWER}"

                    # After first reading, switch to using POLL_INTERVAL
                    if $FIRST_POWER_READING; then
                        POWER_DIVISOR="$POLL_INTERVAL"
                        FIRST_POWER_READING=false
                    fi
                else
                    OUTPUT+="\t-"
                fi

                PREV_RAPL_ENERGY="$CURR_RAPL_ENERGY"
                ;;
            *)
                if $POWER_HAS_CCD; then
                    OUTPUT+="\t-\t-\t-"
                else
                    OUTPUT+="\t-"
                fi
                ;;
        esac
    fi

    # 6. Frequencies (if collecting)
    if $COLLECT_CLOCKS; then
        FREQS=$(get_frequencies)
        OUTPUT+="\t${FREQS}"
    fi

    # 7. Print Data
    echo -e "$OUTPUT"

    # 8. Anti-Drift Sleep (aligns to interval boundaries)
    # Uses bash EPOCHREALTIME (5.0+) when available, falls back to date
    if [[ -n "$EPOCHREALTIME" ]]; then
        sleep $(awk -v now="$EPOCHREALTIME" -v interval="$POLL_INTERVAL" 'BEGIN {printf "%.6f", interval - (now % interval)}')
    else
        sleep $(date +%s.%N | awk -v interval="$POLL_INTERVAL" '{print interval - ($1 % interval)}')
    fi
done
