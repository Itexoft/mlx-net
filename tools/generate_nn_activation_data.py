#!/usr/bin/env python3
"""
Generate deterministic activation test data using Python MLX.

The script mirrors a subset of the MLX Swift integration tests focussed on
parameter-free activation layers. It produces a JSON payload that can be
consumed by the .NET test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import mlx.core as mx
import mlx.nn as nn


def _serialize_array(array: mx.array) -> Dict[str, Any]:
    """Convert an MLX array into a JSON-serialisable payload."""
    flattened = array.reshape(-1).tolist()
    dtype = str(array.dtype).split(".")[-1]
    shape = list(array.shape)
    return {
        "dtype": dtype,
        "shape": shape,
        "data": flattened,
    }


def _normal_input(shape: Tuple[int, ...], seed: int) -> mx.array:
    """Create a deterministic normal-distributed input tensor."""
    mx.random.seed(seed)
    return mx.random.normal(shape)


def _build_activation_tests() -> Iterable[Dict[str, Any]]:
    """Produce test definitions for selected activation layers."""
    # These layers correspond to parameter-free implementations present in
    # src/MlxNet/Mlx/Nn/Activations.cs.
    layer_configs = [
        {"name": "Sigmoid"},
        {"name": "Tanh"},
        {"name": "ReLU"},
        {"name": "LeakyReLU", "parameters": {"negative_slope": 0.2}},
        {"name": "Softmax", "parameters": {"axis": -1}, "shape": (2, 4)},
        {"name": "SiLU"},
        {"name": "GELU", "alias": "Gelu"},
    ]

    for index, config in enumerate(layer_configs):
        python_name = config["name"]
        layer_name = config.get("alias", python_name)
        params: Dict[str, Any] = dict(config.get("parameters", {}))

        shape = tuple(config.get("shape", (2, 4, 3)))
        seed = 1000 + index
        input_tensor = _normal_input(shape, seed)

        if python_name == "Softmax":
            axis = params.get("axis", -1)
            output_tensor = mx.softmax(input_tensor, axis=axis)
        else:
            layer_ctor = getattr(nn, python_name)
            layer = layer_ctor(**params)
            output_tensor = layer(input_tensor)

        yield {
            "name": f"{layer_name}_{index}",
            "layer": layer_name,
            "parameters": params,
            "input": _serialize_array(input_tensor),
            "output": _serialize_array(output_tensor),
        }


def _serialize_parameters(layer: nn.Module) -> Iterable[Dict[str, Any]]:
    params = layer.parameters()

    def recurse(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                next_path = f"{path}.{key}" if path else key
                yield from recurse(value, next_path)
        else:
            yield path, node

    for path, value in recurse(params, ""):
        yield {
            "path": path,
            "tensor": _serialize_array(value),
            "trainable": True,
        }


def _lists_to_native(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_lists_to_native(item) for item in value]
    if isinstance(value, list):
        return [_lists_to_native(item) for item in value]
    return value


def _build_module_tests() -> Iterable[Dict[str, Any]]:
    module_configs = [
        {
            "layer": "Linear",
            "settings": {"inputDimensions": 6, "outputDimensions": 4, "bias": True},
            "input_shape": (3, 6),
        },
        {
            "layer": "Linear",
            "settings": {"inputDimensions": 5, "outputDimensions": 2, "bias": False},
            "input_shape": (1, 5),
        },
        {
            "layer": "Conv1d",
            "settings": {
                "in_channels": 3,
                "out_channels": 2,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "bias": True,
            },
            "input_shape": (2, 8, 3),
        },
        {
            "layer": "Conv2d",
            "settings": {
                "in_channels": 2,
                "out_channels": 3,
                "kernel_size": (3, 3),
                "stride": (1, 1),
                "padding": (1, 1),
                "bias": True,
            },
            "input_shape": (2, 6, 6, 2),
        },
    ]

    for index, config in enumerate(module_configs):
        layer_name = config["layer"]
        settings = {k: _lists_to_native(v) for k, v in config["settings"].items()}
        input_shape = tuple(config["input_shape"])

        seed = 2000 + index
        mx.random.seed(seed)

        layer_ctor = getattr(nn, layer_name)

        if layer_name == "Linear":
            layer = layer_ctor(
                settings["inputDimensions"],
                settings["outputDimensions"],
                bias=settings.get("bias", True),
            )
        elif layer_name == "Conv1d":
            layer = layer_ctor(
                settings["in_channels"],
                settings["out_channels"],
                settings["kernel_size"],
                stride=settings.get("stride", 1),
                padding=settings.get("padding", 0),
                dilation=settings.get("dilation", 1),
                groups=settings.get("groups", 1),
                bias=settings.get("bias", True),
            )
        elif layer_name == "Conv2d":
            layer = layer_ctor(
                settings["in_channels"],
                settings["out_channels"],
                tuple(settings["kernel_size"]),
                stride=tuple(settings.get("stride", (1, 1))),
                padding=tuple(settings.get("padding", (0, 0))),
                dilation=tuple(settings.get("dilation", (1, 1))),
                groups=settings.get("groups", 1),
                bias=settings.get("bias", True),
            )
        else:
            layer = layer_ctor(**settings)

        input_tensor = mx.random.normal(input_shape)
        output_tensor = layer(input_tensor)

        parameters = list(_serialize_parameters(layer))

        yield {
            "name": f"{layer_name}_{index}",
            "layer": layer_name,
            "settings": settings,
            "input": _serialize_array(input_tensor),
            "output": _serialize_array(output_tensor),
            "parameters": parameters,
        }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    activation_dir = repo_root / "src" / "MlxNet.Tests" / "TestData" / "IntegrationActivations"
    activation_dir.mkdir(parents=True, exist_ok=True)

    activation_payload = {"tests": list(_build_activation_tests())}
    activation_path = activation_dir / "activations.json"
    with activation_path.open("w", encoding="utf-8") as handle:
        json.dump(activation_payload, handle, indent=2)
    print(f"Wrote activation test data to {activation_path}")

    modules_dir = repo_root / "src" / "MlxNet.Tests" / "TestData" / "IntegrationModules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    modules_payload = {"tests": list(_build_module_tests())}
    modules_path = modules_dir / "modules.json"
    with modules_path.open("w", encoding="utf-8") as handle:
        json.dump(modules_payload, handle, indent=2)
    print(f"Wrote module test data to {modules_path}")


if __name__ == "__main__":
    main()
