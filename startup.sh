#!/bin/bash
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 ffmpeg
pip install -r requirements.txt
streamlit run main.py --server.port=8000 --server.address=0.0.0.0