import types

import torch
import torch.nn as nn


def _ensure_params_buffers_float32(module: nn.Module) -> None:
    for parameter in module.parameters(recurse=True):
        if parameter.dtype is not torch.float32:
            parameter.data = parameter.data.float()
    for name, buffer in module.named_buffers(recurse=True):
        # Keep buffers as-is unless they are floating; cast float buffers to float32 for stability
        if torch.is_floating_point(buffer) and buffer.dtype is not torch.float32:
            module.register_buffer(name, buffer.float(), persistent=module._is_buffer_persistent(name))


def Float32Module(
    wrapped_module: nn.Module,
    cast_outputs_back: bool = False,
) -> nn.Module:
    """Return the same module but:
    - ensure all parameters (and floating buffers) are stored in float32
    - patch `.to(...)` so it only moves device and ignores dtype conversions
    - optionally cast inputs to float32 for compute and cast outputs back to a reference input dtype
      (disabled by default to avoid interfering with modules that intentionally change dtypes)
    The returned object preserves the original class and attributes.
    """

    def _map_tensors(nested, map_fn):
        if isinstance(nested, torch.Tensor):
            return map_fn(nested)
        if isinstance(nested, tuple):
            return tuple(_map_tensors(x, map_fn) for x in nested)
        if isinstance(nested, list):
            return [_map_tensors(x, map_fn) for x in nested]
        if isinstance(nested, dict):
            return {k: _map_tensors(v, map_fn) for k, v in nested.items()}
        return nested

    def _first_tensor_dtype(args, kwargs):
        for item in args:
            if isinstance(item, torch.Tensor) and torch.is_floating_point(item):
                return item.dtype
        for _, value in kwargs.items():
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                return value.dtype
        return None

    _ensure_params_buffers_float32(wrapped_module)

    original_to = wrapped_module.to

    def to_only_device(self, *args, **kwargs):
        device = None

        if args:
            first = args[0]
            if isinstance(first, (torch.device, str)) or first is None:
                device = first
            elif isinstance(first, torch.dtype):
                if len(args) >= 2 and isinstance(args[1], (torch.device, str)):
                    device = args[1]
            elif isinstance(first, torch.Tensor):
                device = first.device

        if "device" in kwargs:
            device = kwargs["device"]

        if device is not None:
            return original_to(device)
        return self

    wrapped_module.to = types.MethodType(to_only_device, wrapped_module)

    if cast_outputs_back:
        original_forward = wrapped_module.forward

        def forward_float32_then_cast_back(self, *args, **kwargs):
            ref_dtype = _first_tensor_dtype(args, kwargs)
            args_f32 = _map_tensors(args, lambda t: t.float())
            kwargs_f32 = {k: _map_tensors(v, lambda t: t.float()) for k, v in kwargs.items()}
            outputs = original_forward(*args_f32, **kwargs_f32)
            if ref_dtype is None:
                return outputs
            return _map_tensors(outputs, lambda t: t.to(ref_dtype))

        wrapped_module.forward = types.MethodType(forward_float32_then_cast_back, wrapped_module)

    return wrapped_module
