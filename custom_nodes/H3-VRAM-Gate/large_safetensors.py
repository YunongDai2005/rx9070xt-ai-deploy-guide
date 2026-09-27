"""
Workaround: safetensors 0.8.0 safe_open(framework="pt") crashes with a Windows
access violation (0xC0000005) on any get_tensor() of files larger than ~21 GiB
(reproduced with a synthetic 23 GiB file; 20 GiB works). The H3 Qwen3-VL 32B
int8_convrot text encoder is 25.3 GiB, so CLIPLoader kills ComfyUI.

Only comfy.utils' reference is swapped. Files above LIMIT are read through a
read-only Python mmap + torch.frombuffer; everything else still goes to the real
safetensors.safe_open. Read-only file-backed pages don't add commit charge.
"""

import json
import logging
import mmap
import os
import struct
import types
import warnings

import safetensors
import safetensors.torch
import torch

import comfy.utils

log = logging.getLogger("H3-VRAM-Gate")

LIMIT = 16 * 1024 ** 3


class MMapSafeOpen:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        header_size = struct.unpack("<Q", self.mm[:8])[0]
        self.header = json.loads(self.mm[8:8 + header_size])
        self.meta = self.header.pop("__metadata__", None)
        self.base = 8 + header_size
        self.mv = memoryview(self.mm)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Tensors keep views into the mmap, so it must stay open.
        return False

    def keys(self):
        return sorted(self.header)

    def metadata(self):
        return self.meta

    def get_tensor(self, name):
        info = self.header[name]
        start, end = info["data_offsets"]
        dtype = comfy.utils._TYPES[info["dtype"]]
        if start == end:
            return torch.empty(info["shape"], dtype=dtype)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given buffer is not writable")
            return torch.frombuffer(self.mv[self.base + start:self.base + end], dtype=dtype).view(info["shape"])


def safe_open(path, framework="pt", device="cpu"):
    if framework == "pt" and device == "cpu" and os.path.getsize(path) > LIMIT:
        log.info("[H3-VRAM-Gate] mmap loading %s", os.path.basename(path))
        return MMapSafeOpen(path)
    return safetensors.safe_open(path, framework=framework, device=device)


comfy.utils.safetensors = types.SimpleNamespace(safe_open=safe_open, torch=safetensors.torch)
