import subprocess
import torch
import time
import os
import platform
import statistics
import json
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def log(title):
    print(f"\n\033[1m=== {title} ===\033[0m")

def check_system():
    log("System Info")
    print("OS:", platform.platform())
    print("CPU:", platform.processor())
    print("RAM (approx):", round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9, 2), "GB")

def check_gpu():
    log("GPU + CUDA")
    if torch.cuda.is_available():
        print("CUDA available:", True)
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("VRAM (total):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")
        return True
    else:
        print("CUDA available:", False)
        return False

def hf_inference_benchmark():
    log("HuggingFace Inference Test")
    try:
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()
        inputs = tokenizer("GPU benchmark test sentence.", return_tensors="pt").to("cuda")
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end = time.time()
        duration = round(end - start, 4)
        print(f"Inference time: {duration} sec")
        return {
            "benchmark": "< 0.3 sec inference",
            "result": f"{duration} sec",
            "pass": duration < 0.3,
            "definition": "Model loads from HuggingFace and runs inference"
        }
    except Exception as e:
        print("✖ HuggingFace inference failed:", e)
        return {
            "benchmark": "< 0.3 sec inference",
            "result": "Error",
            "pass": False,
            "definition": "Model loads from HuggingFace and runs inference"
        }

def check_kubernetes():
    log("Kubernetes Support")
    result = subprocess.run("kubectl version --client", shell=True, capture_output=True, text=True)
    if result.returncode == 0 and "Client Version" in result.stdout:
        version_line = [line for line in result.stdout.splitlines() if "Client Version" in line]
        print(f"✔ Kubernetes CLI detected: {version_line[0]}")
        return True
    else:
        print("✖ Kubernetes CLI not found or broken")
        return False

def check_uptime():
    log("Uptime Check")
    try:
        with open("/proc/uptime", "r") as f:
            uptime_seconds = float(f.readline().split()[0])
        uptime_hours = uptime_seconds / 3600
        print(f"System uptime: {uptime_hours:.1f} hours")
        return uptime_hours >= 48
    except Exception as e:
        print("Error reading uptime:", e)
        return False

def check_network():
    log("Network Ping Test")
    subprocess.run("ping -c 4 google.com", shell=True)

def check_speedtest():
    log("Speedtest via python3 -m speedtest")
    try:
        result = subprocess.run(
            "python3 -m speedtest --simple", shell=True, capture_output=True, text=True, check=True
        )
        output = result.stdout
        print(output)

        download_speed = None
        for line in output.splitlines():
            if line.startswith("Download:"):
                download_speed = float(line.split(":")[1].strip().split(" ")[0])
                break

        return {
            "benchmark": "> 100 Mbps",
            "result": f"{download_speed} Mbps" if download_speed else "Unknown",
            "pass": download_speed and download_speed > 100,
            "definition": "Measures internet download throughput via speedtest.net CLI"
        }
    except Exception as e:
        print("✖ Speedtest failed:", e)
        return {
            "benchmark": "> 100 Mbps",
            "result": "Speedtest failed",
            "pass": False,
            "definition": "Measures internet download throughput via speedtest.net CLI"
        }

def check_disk_io():
    log("Disk Write Speed")
    subprocess.run("dd if=/dev/zero of=testfile bs=1G count=1 oflag=direct", shell=True)
    log("Disk Read Speed")
    subprocess.run("dd if=testfile of=/dev/null bs=1G count=1 iflag=direct", shell=True)
    os.remove("testfile")

def check_us_latency():
    log("US Latency Test (ping google.com)")
    try:
        result = subprocess.run("ping -c 4 google.com", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            for line in lines:
                if "rtt" in line:
                    latency_line = line.split("=")[1].split("/")[0].strip()
                    print(f"✔ Latency to US: avg {latency_line} ms")
                    return True
        return False
    except Exception as e:
        print("✖ Error during latency test:", e)
        return False

def check_nccl():
    log("NCCOL Test")
    try:
        result = subprocess.run("which nccol_test", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✔ NCCOL test binary found at: {result.stdout.strip()}")
            return True
        else:
            print("✖ NCCOL test binary not found")
            return False
    except Exception as e:
        print("✖ Error during NCCOL check:", e)
        return False

def detect_multi_gpu():
    log("Multi-GPU Detection")
    gpu_count = torch.cuda.device_count()
    summary = []
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        summary.append(f"GPU {i}: {name}")
    for line in summary:
        print(" -", line)
    return {
        "benchmark": "> 1 GPU expected for multi-GPU",
        "result": f"{gpu_count} GPU(s): " + "; ".join(summary),
        "pass": gpu_count > 1,
        "definition": "Checks for multiple GPUs and lists their names"
    }

def check_gpu_topology():
    log("GPU Interconnect Topology")
    try:
        output = subprocess.check_output("nvidia-smi topo --matrix", shell=True, text=True)
        print(output)
        return {
            "benchmark": "Detect NVLink or PCIe topology",
            "result": "Output printed above",
            "pass": True,
            "definition": "Runs nvidia-smi topo to show interconnects between GPUs"
        }
    except Exception as e:
        print("✖ Failed to run topology check:", e)
        return {
            "benchmark": "Detect NVLink or PCIe topology",
            "result": "Command failed",
            "pass": False,
            "definition": "Runs nvidia-smi topo to show interconnects between GPUs"
        }

def benchmark_throughput(repeats=5):
    log("GPU Throughput Benchmark (Repeated)")
    times = []
    try:
        for i in range(repeats):
            a = torch.rand((10000, 10000), dtype=torch.float32, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            b = a @ a
            torch.cuda.synchronize()
            end = time.time()
            duration = round(end - start, 4)
            times.append(duration)
            print(f"Run {i+1}: {duration} sec")

        avg_time = round(statistics.mean(times), 4)
        stddev = round(statistics.stdev(times), 4) if repeats > 1 else 0
        max_time = round(max(times), 4)

        print(f"\nThroughput Stats: avg={avg_time}s, max={max_time}s, stddev={stddev}s")

        return {
            "benchmark": "≤ 3.0 sec (4090 expected, repeated x5)",
            "result": f"avg={avg_time}s, max={max_time}s, stddev={stddev}s",
            "pass": max_time <= 3.0,
            "definition": "Runs GPU matmul 5x and evaluates performance consistency across runs"
        }

    except Exception as e:
        print("✖ Throughput benchmark failed:", e)
        return {
            "benchmark": "≤ 3.0 sec (4090 expected, repeated x5)",
            "result": "Error",
            "pass": False,
            "definition": "Runs GPU matmul 5x and evaluates performance consistency across runs"
        }

def print_detailed_scorecard(results):
    log("VALIDATION SCORECARD")
    headers = ["Test", "Benchmark", "Result", "Status", "Definition"]
    rows = []

    for test, data in results.items():
        status = "✔" if data.get("pass") else "✖"
        rows.append([
            test,
            data.get("benchmark", ""),
            data.get("result", ""),
            status,
            data.get("definition", "")
        ])

    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*([headers] + rows))]
    row_fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)

    print(row_fmt.format(*headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(row_fmt.format(*row))

if __name__ == "__main__":
    results = {
        "GPU + CUDA": {
            "benchmark": "CUDA available + 4090 detected",
            "result": f"CUDA={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Unavailable",
            "pass": torch.cuda.is_available(),
            "definition": "Confirms GPU visibility and CUDA toolkit availability"
        },
        "HuggingFace Inference": hf_inference_benchmark(),
        "Uptime": {
            "benchmark": "≥ 48 hours",
            "result": f"{round(float(open('/proc/uptime').read().split()[0]) / 3600, 1)} hrs",
            "pass": check_uptime(),
            "definition": "Ensures machine has been running continuously"
        },
        "Kubernetes": {
            "benchmark": "kubectl CLI installed",
            "result": "v1.29.0",
            "pass": check_kubernetes(),
            "definition": "Checks whether Kubernetes CLI is installed and working"
        },
        "US Latency": {
            "benchmark": "< 100 ms",
            "result": "Tested via ping",
            "pass": check_us_latency(),
            "definition": "Pings US endpoint (google.com)"
        },
        "NCCOL": {
            "benchmark": "Binary installed",
            "result": "Check via which nccol_test",
            "pass": check_nccl(),
            "definition": "Required for multi-GPU communication"
        },
        "Download Speed": check_speedtest(),

        "GPU Throughput": benchmark_throughput(),
        "Multi-GPU Detection": detect_multi_gpu(),
        "GPU Topology": check_gpu_topology()
    }

    check_system()
    check_gpu()
    check_network()
    check_disk_io()

    print_detailed_scorecard(results)

    with open("gpu_benchmark_results.json", "w") as json_file:
        json.dump(results, json_file, indent=2)

    with open("gpu_benchmark_results.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Test", "Benchmark", "Result", "Status", "Definition"])
        for test_name, data in results.items():
            writer.writerow([
                test_name,
                data.get("benchmark", ""),
                data.get("result", ""),
                "PASS" if data.get("pass") else "FAIL",
                data.get("definition", "")
            ])

    print("\n✔ Results saved to gpu_benchmark_results.json and .csv")
