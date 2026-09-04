#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Python/GPU environment for CA-HNM experiments.")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="Visible CUDA device to test. Under Slurm, the assigned physical GPU is normally remapped to cuda:0.",
    )
    parser.add_argument("--out", default="runs/paper_experiments/environment.json")
    args = parser.parse_args()

    try:
        import sentence_transformers
        import torch
    except Exception as exc:
        raise RuntimeError("torch and sentence-transformers must be installed") from exc

    cuda_available = torch.cuda.is_available()
    cuda_devices = []
    cuda_runtime_errors = []
    try:
        torch_cuda_arch_list = torch.cuda.get_arch_list() if cuda_available else []
    except Exception as exc:
        torch_cuda_arch_list = []
        cuda_runtime_errors.append(f"unable to read compiled CUDA architectures: {type(exc).__name__}: {exc}")
    visible_cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    if cuda_available:
        if not 0 <= args.cuda_device < visible_cuda_device_count:
            raise ValueError(
                f"--cuda-device={args.cuda_device} is outside the visible device range "
                f"0..{visible_cuda_device_count - 1}"
            )
        for index in [args.cuda_device]:
            try:
                properties = torch.cuda.get_device_properties(index)
                capability = list(torch.cuda.get_device_capability(index))
                device_report = {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "capability": capability,
                    "sm": f"sm_{capability[0]}{capability[1]}",
                }
                try:
                    with torch.cuda.device(index):
                        value = torch.ones((8, 8), device=f"cuda:{index}")
                        result = value @ value
                        torch.cuda.synchronize(index)
                        device_report["tensor_smoke_test"] = bool(result[0, 0].item() == 8.0)
                except Exception as exc:
                    device_report["tensor_smoke_test"] = False
                    error = f"cuda:{index} tensor test failed: {type(exc).__name__}: {exc}"
                    device_report["error"] = error
                    cuda_runtime_errors.append(error)
                cuda_devices.append(device_report)
            except Exception as exc:
                error = f"cuda:{index} inspection failed: {type(exc).__name__}: {exc}"
                cuda_devices.append({"index": index, "error": error, "tensor_smoke_test": False})
                cuda_runtime_errors.append(error)

    cudnn_version = None
    cudnn_error = None
    if cuda_available:
        try:
            cudnn_version = torch.backends.cudnn.version()
        except Exception as exc:
            cudnn_error = f"{type(exc).__name__}: {exc}"
            cuda_runtime_errors.append(f"cuDNN initialization failed: {cudnn_error}")
    disk = shutil.disk_usage(Path.cwd())
    report = {
        "schema_version": 2,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "cuda_available": cuda_available,
        "visible_cuda_device_count": visible_cuda_device_count,
        "tested_cuda_device": args.cuda_device if cuda_available else None,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_arch_list": torch_cuda_arch_list,
        "cudnn_version": cudnn_version,
        "cudnn_error": cudnn_error,
        "cuda_runtime_errors": cuda_runtime_errors,
        "cuda_devices": cuda_devices,
        "workspace": str(Path.cwd()),
        "disk_free_bytes": disk.free,
        "checks": {
            "python_supported": sys.version_info >= (3, 10),
            "cuda_requirement_satisfied": (
                cuda_available and not cuda_runtime_errors
            ) or not args.require_cuda,
            "disk_free_at_least_20gb": disk.free >= 20 * 1024**3,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all(report["checks"].values()):
        if args.require_cuda and cuda_runtime_errors:
            print(
                "CUDA is visible but unusable. For Tesla V100/Volta (sm_70), install the pinned "
                "CUDA 12.6 wheel with: bash scripts/server/bootstrap.sh",
                file=sys.stderr,
            )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
