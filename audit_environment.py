import hashlib
import importlib.metadata
import json
import pathlib
import subprocess
import requests
import torch

ROOT = pathlib.Path(__file__).parent
repo = 'Comfy-Org/Qwen-Image-2.1'
r = requests.get('https://huggingface.co/api/models/' + repo + '?blobs=true', timeout=45)
r.raise_for_status()
j = r.json()
files = []
for name in ['diffusion_models/qwen_image_2.1_int8_convrot.safetensors',
             'text_encoders/qwen3vl_8b_int8_convrot.safetensors',
             'vae/qwen_image_2.1_vae_bf16.safetensors']:
    meta = next(x for x in j['siblings'] if x['rfilename'] == name)
    p = pathlib.Path('E:/AI/Models/QwenImage21') / name
    with p.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    files.append(dict(path=str(p), bytes=p.stat().st_size, sha256=digest,
                      official_sha256=meta['lfs']['sha256'], matches=digest==meta['lfs']['sha256'],
                      url=f'https://huggingface.co/{repo}/resolve/{j["sha"]}/{name}'))
    print(p.name, 'official hash match:', files[-1]['matches'], flush=True)
packages = ['torch','torchvision','torchaudio','gguf','transformers','comfy-kitchen','comfy-aimdo',
            'comfyui-frontend-package','safetensors','sentencepiece','protobuf']
env = dict(packages={p:importlib.metadata.version(p) for p in packages},
           hip=torch.version.hip, gpu=str(torch.cuda.get_device_properties(0)), files=files,
           commits={p:subprocess.check_output(['git','-C',p,'rev-parse','HEAD'],text=True).strip()
                    for p in ['E:/AI/Apps/ComfyUI','E:/AI/Apps/ComfyUI/custom_nodes/ComfyUI-GGUF']})
(ROOT / 'environment-audit.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
