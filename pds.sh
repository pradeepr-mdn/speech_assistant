#!/bin/bash
cd /home/mdn/Desktop/work/speech-assistant || exit
source assistant_env/bin/activate
python3 yolo-azure.py
