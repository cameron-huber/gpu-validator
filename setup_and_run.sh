#!/bin/bash

echo "[*] Installing required Python packages..."
pip3 install --upgrade torch transformers speedtest-cli

echo "[*] Downloading GPU validator..."
curl -O https://yourdomain.com/gpu_validate.py

echo "[*] Running validator..."
python3 gpu_validate.py
