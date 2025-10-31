#!/bin/bash
cd /home/mdn/sens-able-prama/speech-assistant || exit
source models_env/bin/activate
python3 yolo-azure.py
