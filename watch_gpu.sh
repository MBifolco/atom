#!/bin/bash
# GPU monitoring script for training

while true; do
    clear
    echo "========================================"
    echo "  GPU Monitor - $(date '+%H:%M:%S')"
    echo "========================================"
    echo

    # Get GPU info from rocm-smi
    GPU_INFO=$(rocm-smi)

    # Extract key metrics
    GPU_UTIL=$(echo "$GPU_INFO" | grep "GPU use" | awk '{print $4}')
    TEMP=$(echo "$GPU_INFO" | grep "Temperature" | awk '{print $3}')
    POWER=$(echo "$GPU_INFO" | grep "Average Graphics Package Power" | awk '{print $5}')

    # Get memory info
    MEM_INFO=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM Total Memory")
    MEM_USED=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM Total Used Memory" | awk '{print $6}')
    MEM_TOTAL=$(echo "$MEM_INFO" | awk '{print $6}')

    if [ -n "$MEM_USED" ] && [ -n "$MEM_TOTAL" ]; then
        MEM_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($MEM_USED/$MEM_TOTAL)*100}")
    else
        MEM_PERCENT="N/A"
    fi

    echo "📊 GPU Utilization: ${GPU_UTIL:-N/A}"
    echo "🌡️  Temperature:     ${TEMP:-N/A}°C"
    echo "⚡ Power:           ${POWER:-N/A} W"
    echo "💾 Memory Used:     ${MEM_USED:-N/A} MB / ${MEM_TOTAL:-N/A} MB (${MEM_PERCENT}%)"
    echo

    # Show process info
    echo "🔥 GPU Processes:"
    ps aux | grep -E "(python.*train|python.*resume)" | grep -v grep | while read line; do
        PID=$(echo "$line" | awk '{print $2}')
        CMD=$(echo "$line" | awk '{print $11" "$12" "$13}')
        echo "  PID $PID: $CMD"
    done

    echo
    echo "Press Ctrl+C to exit"

    sleep 2
done
