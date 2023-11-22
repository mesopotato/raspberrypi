#!/bin/bash
sleep 30
# Define the temperature threshold (in degrees Celsius)
TEMP_THRESHOLD=75

# Define the interval for checking the temperature (in seconds)
CHECK_INTERVAL=300

# Define the local and NAS log file paths
local_log="/home/yannick/classify_log.txt"
nas_log="/home/yannick/fileServer/classify_log.txt"

# Loop indefinitely
while true; do
    # Get the current temperature
    temp=$(vcgencmd measure_temp | egrep -o '[0-9.]+')
    echo "Current temperature: $temp°C" | tee -a $local_log $nas_log
    # Check if the temperature exceeds the threshold
    if (( $(echo "$temp > $TEMP_THRESHOLD" | bc -l) )); then
        echo "Device is overheating at ${temp}°C, stopping classify.py..." | tee -a $local_log $nas_log

        # Kill the classify.py process
        pkill -f classify.py

    fi

    # Wait for the specified interval before checking again
    sleep $CHECK_INTERVAL
done
