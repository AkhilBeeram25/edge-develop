"""TensorRT command builder for exported YOLO-Micro ONNX models."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex


def build_trtexec_command(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
    int8: bool = False,
    workspace_mb: int = 4096,
) -> list[str]:
    command = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mb}",
    ]
    if fp16:
        command.append("--fp16")
    if int8:
        command.append("--int8")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--workspace-mb", type=int, default=4096)
    args = parser.parse_args()
    command = build_trtexec_command(
        args.onnx,
        args.engine,
        fp16=not args.no_fp16,
        int8=args.int8,
        workspace_mb=args.workspace_mb,
    )
    print(shlex.join(command))


if __name__ == "__main__":
    main()

