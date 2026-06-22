"""Quantization-aware training helpers."""

from __future__ import annotations


def prepare_qat_model(model: object, backend: str = "fbgemm") -> object:
    """Prepare a PyTorch model for QAT while keeping call sites explicit."""

    try:
        import torch
        from torch.ao.quantization import get_default_qat_qconfig, prepare_qat
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("QAT preparation requires PyTorch.") from exc

    torch.backends.quantized.engine = backend
    model.train()  # type: ignore[attr-defined]
    model.qconfig = get_default_qat_qconfig(backend)  # type: ignore[attr-defined]
    return prepare_qat(model, inplace=False)


def convert_qat_model(model: object) -> object:
    try:
        from torch.ao.quantization import convert
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("QAT conversion requires PyTorch.") from exc
    model.eval()  # type: ignore[attr-defined]
    return convert(model, inplace=False)

