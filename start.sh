#!/bin/bash
echo "Activating virtual environment"
source /home/yannick/Documents/tensor/env/bin/activate
echo "Changing directory"
cd /home/yannick/Documents/tensor/examples/lite/examples/image_classification/raspberry_pi
sleep 20
while true; do
	echo "Starting Python script"
	python3 classify.py >> /home/yannick/classify_log.txt 2>&1
	echo "Script execution interrupted. restarting in 5 seconds.."
	sleep 5
done

