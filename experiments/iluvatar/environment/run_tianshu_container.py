#!/usr/bin/env python3
"""Launch an isolated, disposable MiniCPM competition container on ub39."""

import argparse
import os
import re
import subprocess
import sys


ROOT = "/workspace/wyNomad"
IMAGE = (
    "harbor.baai.ac.cn/plugin/"
    "iluvatar-corex4.5.0-flagtree0.6.0-triton3.6.0-cxnone-vllm_fl0.24.0:20260909"
)
DRIVER = "/usr/local/corex-4.5.0"
IDLE_LIMIT_MIB = 256


def gpu_memory_used(gpu_id):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = DRIVER + "/lib64:" + env.get("LD_LIBRARY_PATH", "")
    result = subprocess.run(
        ["/usr/local/corex/bin/ixsmi"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    lines = result.stdout.splitlines()
    card = re.compile(r"^\|\s*" + str(gpu_id) + r"\s+Iluvatar BI-V150\b")
    for index, line in enumerate(lines[:-1]):
        if card.search(line):
            memory = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB", lines[index + 1])
            if memory:
                return int(memory.group(1)), int(memory.group(2))
    raise RuntimeError("无法读取 GPU {} 的显存状态".format(gpu_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detach", action="store_true", help="后台启动，供服务和评测分开执行")
    parser.add_argument("gpu_id", type=int, nargs="?", default=0, help="宿主机 GPU 编号，默认 0")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="可选：-- 后跟容器内命令")
    args = parser.parse_args()
    if not 0 <= args.gpu_id <= 15:
        parser.error("GPU 编号必须在 0 到 15 之间")

    paths = [
        ROOT + "/vllm-plugin-FL",
        ROOT + "/FlagGems",
        ROOT + "/results",
        ROOT + "/logs",
        ROOT + "/evalscope-results",
        "/workspace/MiniCPM5-2B",
        "/workspace/evalscope-datasets/math_500",
        DRIVER,
        "/dev/iluvatar{}".format(args.gpu_id),
        "/dev/itrctl",
        "/dev/itrlink",
        "/dev/itr_peerm_dev0",
    ]
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        parser.error("缺少路径：" + ", ".join(missing))

    try:
        used, total = gpu_memory_used(args.gpu_id)
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        parser.error("无法检查 GPU：{}".format(exc))
    if used > IDLE_LIMIT_MIB:
        parser.error(
            "GPU {} 已用 {}/{} MiB，超过空闲阈值 {} MiB；请选择另一张卡".format(
                args.gpu_id, used, total, IDLE_LIMIT_MIB
            )
        )

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        command = ["sleep", "infinity"] if args.detach else ["/bin/bash"]

    docker = ["docker", "run", "--rm", "--pull=never"]
    if args.detach:
        docker.append("-d")
    elif sys.stdin.isatty() and sys.stdout.isatty():
        docker.append("-it")
    docker += [
        "--name", "wynomad-minicpm-s2-gpu{}".format(args.gpu_id),
        "--shm-size=16g",
        "--device=/dev/iluvatar{}".format(args.gpu_id),
        "--device=/dev/itrctl",
        "--device=/dev/itrlink",
        "--device=/dev/itr_peerm_dev0",
        "--mount", "type=bind,src={},dst={},readonly".format(DRIVER, DRIVER),
        "--mount", "type=bind,src={0}/vllm-plugin-FL,dst=/workspace/vllm-plugin-FL".format(ROOT),
        "--mount", "type=bind,src={0}/FlagGems,dst=/workspace/FlagGems".format(ROOT),
        "--mount", "type=bind,src=/workspace/MiniCPM5-2B,dst=/workspace/MiniCPM5-2B,readonly",
        "--mount", "type=bind,src=/workspace/evalscope-datasets/math_500,dst=/workspace/evalscope-datasets/math_500,readonly",
        "--mount", "type=bind,src={0}/evalscope-results,dst=/workspace/evalscope-datasets/level3".format(ROOT),
        "--mount", "type=bind,src={0}/results,dst=/workspace/benchmark_results".format(ROOT),
        "--mount", "type=bind,src={0}/logs,dst=/workspace/logs".format(ROOT),
        "-e", "COREX_VISIBLE_DEVICES=0",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        "-e", "VLLM_PLUGINS=fl",
        "-e", "PYTHONPATH=/workspace/vllm-plugin-FL:/workspace/FlagGems/src",
        "-w", "/workspace/vllm-plugin-FL",
        IMAGE,
    ]
    print("启动物理 GPU {}（容器内编号 0，当前已用 {} MiB）".format(args.gpu_id, used), flush=True)
    os.execvp("docker", docker + command)


if __name__ == "__main__":
    main()
