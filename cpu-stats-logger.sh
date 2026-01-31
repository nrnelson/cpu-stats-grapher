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
#   ./cpu-stats-logger.sh | tee output.log   # See output while logging
#
# Requirements:
#   - lm-sensors package installed
#   - AMD CPU with k10temp support
#   - zenergy kernel module (optional, for power monitoring)
#
# Output: Tab-separated values with header row
# Stop with Ctrl+C
#
# --- CONFIGURATION ---
# Adjust these values for your system. Run 'sensors' to find your sensor name.
SENSOR="k10temp-pci-00c3"
INPUT="temp1_input"
TEMP_LIMIT=89.0

# zenergy sensor for power monitoring (auto-detect hwmon path)
ZENERGY_HWMON=$(grep -l "zenergy" /sys/class/hwmon/hwmon*/name 2>/dev/null | head -1 | xargs dirname)
# ---------------------

# Initialize variables for CPU % Calculation
PREV_TOTAL=0
PREV_IDLE=0

# Initialize variables for Power Calculation (zenergy gives cumulative energy in mJ)
PREV_SOCKET_ENERGY=0
PREV_CCD0_ENERGY=0
PREV_CCD1_ENERGY=0

# Dynamic Header
CORE_COUNT=$(nproc)
HEADER="timestamp\ttemp_c\tload_avg\tusage_pct\tthrottled\tpower_w\tccd0_w\tccd1_w"
for ((i=0; i<CORE_COUNT; i++)); do
    HEADER+="\tcore${i}_mhz"
done

echo -e "$HEADER"

while true; do
    # 1. Get Time
    DATE_TIME=$(date +%Y-%m-%d%t%H:%M:%S)

    # 2. Get Load Average (1-minute avg only) from /proc/loadavg
    LOAD_AVG=$(awk '{print $1}' /proc/loadavg)

    # 3. Calculate CPU Usage %
    read -r cpu user nice system idle iowait irq softirq steal guest rest < /proc/stat

    TOTAL=$((user + nice + system + idle + iowait + irq + softirq + steal))

    DIFF_IDLE=$((idle - PREV_IDLE))
    DIFF_TOTAL=$((TOTAL - PREV_TOTAL))

    if [ "$DIFF_TOTAL" -ne 0 ]; then
        CPU_USAGE=$(( (100 * (DIFF_TOTAL - DIFF_IDLE)) / DIFF_TOTAL ))
    else
        CPU_USAGE=0
    fi

    PREV_TOTAL="$TOTAL"
    PREV_IDLE="$idle"

    # 4. Get Temp
    TEMP_C=$(sensors -u "$SENSOR" | awk -v input="$INPUT" '$1 == input ":" {print $2}')

    # 5. Detect Throttling
    IS_THROTTLED=$(awk -v t="$TEMP_C" -v l="$TEMP_LIMIT" 'BEGIN {print (t >= l - 0.5) ? "YES" : "-"}')

    # 6. Get Power from zenergy (energy counters in microjoules)
    if [ -n "$ZENERGY_HWMON" ]; then
        # Socket total is energy17, cores 0-7 (CCD0) are energy1-8, cores 8-15 (CCD1) are energy9-16
        CURR_SOCKET_ENERGY=$(cat "$ZENERGY_HWMON/energy17_input" 2>/dev/null || echo 0)

        # Sum CCD0 (cores 0-7) and CCD1 (cores 8-15) energy
        CCD0_ENERGY=0
        CCD1_ENERGY=0
        for i in {1..8}; do
            val=$(cat "$ZENERGY_HWMON/energy${i}_input" 2>/dev/null || echo 0)
            CCD0_ENERGY=$((CCD0_ENERGY + val))
        done
        for i in {9..16}; do
            val=$(cat "$ZENERGY_HWMON/energy${i}_input" 2>/dev/null || echo 0)
            CCD1_ENERGY=$((CCD1_ENERGY + val))
        done

        if [ "$PREV_SOCKET_ENERGY" -ne 0 ]; then
            # Delta in microjoules, convert to Watts (uJ / 1000000 = J, J/s = W)
            DELTA_SOCKET=$((CURR_SOCKET_ENERGY - PREV_SOCKET_ENERGY))
            DELTA_CCD0=$((CCD0_ENERGY - PREV_CCD0_ENERGY))
            DELTA_CCD1=$((CCD1_ENERGY - PREV_CCD1_ENERGY))

            # Handle counter wraparound (unlikely but safe)
            [ "$DELTA_SOCKET" -lt 0 ] && DELTA_SOCKET=0
            [ "$DELTA_CCD0" -lt 0 ] && DELTA_CCD0=0
            [ "$DELTA_CCD1" -lt 0 ] && DELTA_CCD1=0

            SOCKET_POWER=$(awk "BEGIN {printf \"%.1f\", $DELTA_SOCKET / 1000000}")
            CCD0_POWER=$(awk "BEGIN {printf \"%.1f\", $DELTA_CCD0 / 1000000}")
            CCD1_POWER=$(awk "BEGIN {printf \"%.1f\", $DELTA_CCD1 / 1000000}")
        else
            SOCKET_POWER="-"
            CCD0_POWER="-"
            CCD1_POWER="-"
        fi

        PREV_SOCKET_ENERGY="$CURR_SOCKET_ENERGY"
        PREV_CCD0_ENERGY="$CCD0_ENERGY"
        PREV_CCD1_ENERGY="$CCD1_ENERGY"
    else
        SOCKET_POWER="-"
        CCD0_POWER="-"
        CCD1_POWER="-"
    fi

    # 7. Get Frequencies
    FREQS=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | awk '{printf "%.0f\t", $1/1000}')

    # 8. Print Data
    echo -e "${DATE_TIME}\t${TEMP_C}\t${LOAD_AVG}\t${CPU_USAGE}%\t${IS_THROTTLED}\t${SOCKET_POWER}\t${CCD0_POWER}\t${CCD1_POWER}\t${FREQS}"

    # 9. Anti-Drift Sleep
    sleep $(date +%s.%N | awk '{print 1 - ($1 % 1)}')
done
