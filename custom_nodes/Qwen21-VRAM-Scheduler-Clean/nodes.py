"""
Clean Qwen-Image-2.1 VRAM scheduler for ComfyUI.

One node, one responsibility:
  prompt/image changed:
    1) fully evict other GPU models (notably diffusion)
    2) call ComfyUI's OFFICIAL TextEncodeQwenImage21.execute()
    3) fully offload the text encoder
    4) return conditioning/latent to the sampler

  seed-only changed:
    ComfyUI reuses this node's cached outputs, so no model swap occurs.

No duplicate V1/V2/V3 nodes, no custom reimplementation of Qwen image-edit encoding.
"""

import logging
import torch

import comfy.model_management as mm
from comfy_extras.nodes_qwen import TextEncodeQwenImage21

log = logging.getLogger("Qwen21-VRAM-Scheduler-Clean")

MB = 1024 * 1024


def _free_mb(device):
    try:
        return mm.get_free_memory(device) / MB
    except Exception:
        return -1.0


def _is_same_or_clone(a, b):
    if a is b:
        return True
    try:
        if a.is_clone(b):
            return True
    except Exception:
        pass
    try:
        if b.is_clone(a):
            return True
    except Exception:
        pass
    # Fallback for versions exposing clone_base_uuid directly.
    au = getattr(a, "clone_base_uuid", None)
    bu = getattr(b, "clone_base_uuid", None)
    return au is not None and bu is not None and au == bu


def _loaded_entries_for(patcher):
    out = []
    for lm in list(mm.current_loaded_models):
        try:
            if lm.model is not None and _is_same_or_clone(patcher, lm.model):
                out.append(lm)
        except Exception:
            pass
    return out


def _evict_other_gpu_models(clip_patcher):
    """
    Fully unload GPU-resident models other than this CLIP/text encoder.

    Passing a deliberately huge requirement to ComfyUI's own free_memory()
    avoids the normal 'free only the shortfall' partial-unload behavior that
    previously produced lowvram patches on the diffusion model.
    """
    device = clip_patcher.load_device
    if mm.is_device_cpu(device):
        return

    before = _free_mb(device)
    keep = _loaded_entries_for(clip_patcher)

    with torch.inference_mode():
        unloaded = mm.free_memory(1e30, device, keep_loaded=keep)
        if unloaded:
            mm.soft_empty_cache()

    after = _free_mb(device)
    log.info(
        "[QwenVRAM-CLEAN] Pre-encode eviction: unloaded=%d, free VRAM %.1f -> %.1f MB",
        len(unloaded), before, after
    )


def _offload_text_encoder(clip_patcher):
    """
    Fully unload only this CLIP/text encoder after conditioning has been made.
    The Python model object remains cached, so future prompt changes reload
    weights from RAM/cache rather than requiring this node to recreate the loader.
    """
    device = clip_patcher.load_device
    if mm.is_device_cpu(device):
        return

    before = _free_mb(device)
    indices = []

    snapshot = list(mm.current_loaded_models)
    for i, lm in enumerate(snapshot):
        try:
            if lm.model is not None and _is_same_or_clone(clip_patcher, lm.model):
                indices.append(i)
        except Exception:
            pass

    removed = 0
    with torch.inference_mode():
        for i in sorted(indices, reverse=True):
            if i >= len(mm.current_loaded_models):
                continue
            lm = mm.current_loaded_models[i]
            try:
                if lm.model is None or not _is_same_or_clone(clip_patcher, lm.model):
                    continue
                fully_unloaded = lm.model_unload()
                if fully_unloaded:
                    mm.current_loaded_models.pop(i)
                    removed += 1
            except Exception:
                log.exception("[QwenVRAM-CLEAN] Failed to unload encoder entry")

        if removed:
            mm.soft_empty_cache()

    after = _free_mb(device)
    log.info(
        "[QwenVRAM-CLEAN] Post-encode text encoder unload: entries=%d, free VRAM %.1f -> %.1f MB",
        removed, before, after
    )


class Qwen21ScheduledSingleImageEdit:
    """
    Wrapper around ComfyUI's official TextEncodeQwenImage21 node.

    The prompt and reference image are inputs of THIS SAME node. Therefore:
      - changing prompt/image invalidates this node and runs pre-eviction
      - changing only seed leaves this node cached
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image_1": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": ""
                }),
                "resolution": ("INT", {
                    "default": 992,
                    "min": 0,
                    "max": 4096,
                    "step": 32
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "Qwen Image 2.1/VRAM"
    DESCRIPTION = (
        "Single-image Qwen-Image-2.1 edit encoder with deterministic VRAM swapping. "
        "Uses ComfyUI's official TextEncodeQwenImage21 implementation."
    )

    def encode(self, clip, vae, image_1, prompt, negative_prompt="", resolution=992):
        # Prompt/image changes reach this function, so evict diffusion BEFORE encoding.
        _evict_other_gpu_models(clip.patcher)

        try:
            # IMPORTANT: official node expects one `images` dict, not an `image_1`
            # keyword argument.
            out = TextEncodeQwenImage21.execute(
                clip,
                prompt,
                negative_prompt,
                vae=vae,
                resolution=resolution,
                images={"image_1": image_1},
            )

            # New ComfyUI io.NodeOutput exposes positional results as .args.
            if hasattr(out, "args"):
                positive, negative, latent = out.args[:3]
            else:
                positive, negative, latent = tuple(out)[:3]

        finally:
            # Encoder is not needed by diffusion sampling.
            _offload_text_encoder(clip.patcher)

        log.info(
            "[QwenVRAM-CLEAN] Scheduled edit encode complete; sampler can load diffusion."
        )
        return (positive, negative, latent)


NODE_CLASS_MAPPINGS = {
    "Qwen21ScheduledSingleImageEdit": Qwen21ScheduledSingleImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen21ScheduledSingleImageEdit": "Qwen 2.1 Scheduled Single-Image Edit",
}
