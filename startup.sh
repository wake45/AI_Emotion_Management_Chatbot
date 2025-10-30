#!/bin/bash
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 ffmpeg
pip install --no-cache-dir -r requirements.txt
pip install deepface --no-deps
streamlit run main.py --server.port=8000 --server.address=0.0.0.0 --server.enableCORS=false