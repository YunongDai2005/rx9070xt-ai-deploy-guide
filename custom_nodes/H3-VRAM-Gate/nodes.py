"""
MiniMax H3 VRAM gate: put between SamplerCustomAdvanced and the two VAE decodes.

Fully unloads every GPU model (H3 diffusion, text encoder, VAEs) once sampling is
done, so VideoVAE decodes with the whole card free. Works around the co-residency
slowdown / HIP launch failures seen on 16 GB RDNA4 (ComfyUI issue #15484).
"""

import gc
import logging

import comfy.model_management as mm

log = logging.getLogger("H3-VRAM-Gate")

MB = 1024 * 1024


class H3UnloadBeforeDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    # The latent has to flow through this node so it runs after the sampler
    # and before both decoders.
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "unload"
    CATEGORY = "MiniMax H3/VRAM"
    DESCRIPTION = "Unload all models and empty the VRAM cache before VAE decode."

    def unload(self, latent):
        device = mm.get_torch_device()
        before = mm.get_free_memory(device) / MB
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        log.info("[H3-VRAM-Gate] free VRAM %.0f -> %.0f MB", before, mm.get_free_memory(device) / MB)
        return (latent,)


NODE_CLASS_MAPPINGS = {"H3UnloadBeforeDecode": H3UnloadBeforeDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"H3UnloadBeforeDecode": "H3 Unload Before Decode"}
