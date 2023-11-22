#!/bin/bash
echo "Activating virtual environment" | tee -a $local_log $nas_log
source /home/yannick/Documents/tensor/env/bin/activate
echo "Changing directory" | tee -a $local_log $nas_log
cd /home/yannick/Documents/tensor/examples/lite/examples/image_classification/raspberry_pi
# Define the local and NAS log file paths
local_log="/home/yannick/classify_log.txt"
nas_log="/home/yannick/fileServer/classify_log.txt"
sleep 20
# Infinite loop to restart the script whenever it exits
while true; do
    if [ -f "/home/yannick/fileServer/stop.txt" ]; then
        echo "Stop file detected. Stopping script." | tee -a $local_log $nas_log
        break
    fi

    echo "Starting classify.py..." | tee -a $local_log $nas_log
    python3 classify.py 2>&1 | tee -a $local_log | tee -a $nas_log

    echo "classify.py has stopped. Logging temperature..." | tee -a $local_log $nas_log
    current_time=$(date)
    echo "$current_time: Script stopped. Checking temperature..." | tee -a $local_log $nas_log
    vcgencmd measure_temp | tee -a $local_log $nas_log

    # Wait until the temperature drops below 75°C
    while true; do
        temp=$(vcgencmd measure_temp | egrep -o '[0-9.]+')
        echo "Current temperature: $temp°C" | tee -a $local_log $nas_log
        if (( $(echo "$temp < 75" | bc -l) )); then
            echo "Temperature is below 75°C, restarting script." | tee -a $local_log $nas_log
            break
        fi
        echo "Waiting for the device to cool down..." | tee -a $local_log $nas_log
        sleep 60  # wait for 60 seconds before checking again
    done

    echo "Restarting script in 5 seconds..." | tee -a $local_log $nas_log
    sleep 5
done
