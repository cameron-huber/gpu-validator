#!/bin/bash≈

set -e

echo "[*] Updating package manager..."
sudo apt update

echo "[*] Installing system dependencies (Python3, pip, build tools)..."
sudo apt install -y python3 python3-pip python3-dev build-essential curl git

echo "[*] Installing required Python packages..."
pip3 install --upgrade pip
pip3 install torch transformers speedtest-cli

echo "[*] Downloading GPU validator..."
curl -O https://raw.githubusercontent.com/cameron-huber/gpu-validator/main/gpu_validate.py

echo "[*] Running GPU validator..."
python3 gpu_validate.py
