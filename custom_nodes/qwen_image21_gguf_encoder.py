import logging

import nodes
import comfy.sd
import comfy.text_encoders.qwen_image21
import folder_paths
from safetensors import safe_open


class QwenImage21GGUFEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'clip_name': (folder_paths.get_filename_list('clip_gguf'),),
            'vision_source': (folder_paths.get_filename_list('text_encoders'),),
        }}

    RETURN_TYPES = ('CLIP',)
    FUNCTION = 'load_clip'
    CATEGORY = 'Qwen Image 2.1'

    def load_clip(self, clip_name, vision_source):
        loader = nodes.NODE_CLASS_MAPPINGS['CLIPLoaderGGUF']()
        path = folder_paths.get_full_path_or_raise('clip_gguf', clip_name)
        sd = loader.load_data([path])[0]
        source = folder_paths.get_full_path_or_raise('text_encoders', vision_source)
        # The separate GGUF contains only the language tower. Original visual
        # weights preserve Qwen3-VL detection and avoid uninitialized vision weights.
        with safe_open(source, framework='pt', device='cpu') as f:
            for key in f.keys():
                if key.startswith('model.visual.'):
                    sd[key] = f.get_tensor(key)
        if 'model.visual.deepstack_merger_list.0.norm.weight' not in sd:
            raise ValueError('vision_source must contain the original Qwen3-VL vision tower.')
        clip = loader.load_patcher([path], comfy.sd.CLIPType.QWEN_IMAGE, [sd])
        if not isinstance(clip.tokenizer, comfy.text_encoders.qwen_image21.QwenImage21Tokenizer):
            raise RuntimeError('Expected QwenImage21Tokenizer; refusing a different encoding path.')
        logging.info('QwenImage21GGUFEncoder: Qwen-Image-2.1 tokenizer + Qwen3-VL GGUF language tower + original vision weights')
        return (clip,)


NODE_CLASS_MAPPINGS = {'QwenImage21GGUFEncoder': QwenImage21GGUFEncoder}
NODE_DISPLAY_NAME_MAPPINGS = {'QwenImage21GGUFEncoder': 'Qwen Image 2.1 GGUF Encoder (Original Vision)'}
