# Hyperbolic GPU Validator

This tool benchmarks any GPU node and generates a detailed scorecard including:

- GPU detection (model, VRAM, CUDA)
- HuggingFace inference speed
- Network speed (via speedtest.net)
- Disk I/O performance
- Uptime + Kubernetes + NCCOL
- Multi-GPU detection + topology
- GPU throughput (matmul benchmark)

## Run it on any machine

```bash
curl -sSL https://raw.githubusercontent.com/cameron-huber/gpu-validator/main/setup_and_run.sh | bash
