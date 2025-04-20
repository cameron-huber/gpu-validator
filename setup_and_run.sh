#!/bin/bash
set -e

echo "[*] Updating package manager..."
sudo apt update

echo "[*] Installing base system dependencies..."
sudo apt install -y python3 python3-pip python3-dev build-essential curl git iputils-ping

echo "[*] Installing kubectl (direct download)..."
curl -LO "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl
kubectl version --client || echo "kubectl install failed"

echo "[*] Installing required Python packages..."
pip3 install --upgrade pip
pip3 install --user torch transformers speedtest-cli

# Ensure ~/.local/bin is in PATH so speedtest-cli works
export PATH=$PATH:$HOME/.local/bin

echo "[*] Downloading GPU validator..."
curl -O https://raw.githubusercontent.com/cameron-huber/gpu-validator/main/gpu_validate.py

echo "[*] Running GPU validator..."
python3 gpu_validate.py
