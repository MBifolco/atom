#!/bin/bash
# Find and tail the most recent population training log

# Find the most recent resumed directory
LATEST_DIR=$(ls -td outputs/resumed_* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    # No resumed dir, check progressive dirs
    LATEST_DIR=$(ls -td outputs/progressive_* 2>/dev/null | head -1)
fi

if [ -z "$LATEST_DIR" ]; then
    echo "No training directories found in outputs/"
    exit 1
fi

# Find the log file
LOG_FILE=$(find "$LATEST_DIR" -name "population_training_*.log" -type f | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "No log file found in $LATEST_DIR"
    exit 1
fi

echo "Tailing: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo "----------------------------------------"
tail -f "$LOG_FILE"
