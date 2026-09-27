import json
import pathlib
import time
import torch
import comfy_kitchen as ck
from comfy_kitchen.tensor import QuantizedTensor

ck.registry.disable('cuda')
torch.manual_seed(42)
result = {'torch': torch.__version__, 'hip': torch.version.hip,
          'gpu': str(torch.cuda.get_device_properties(0)),
          'backends': ck.registry.list_backends()}
try:
    x = torch.randn(256, 4096, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16) * 0.02
    q = QuantizedTensor.from_float(w, 'TensorWiseINT8Layout', convrot=True, per_channel=True)
    start = time.perf_counter()
    y = torch.nn.functional.linear(x, q)
    torch.cuda.synchronize()
    ref = torch.nn.functional.linear(x, w)
    result['int8_convrot'] = {'seconds': time.perf_counter()-start,
                             'finite': bool(torch.isfinite(y).all()),
                             'relative_rmse': float(((y.float()-ref.float())**2).mean().sqrt()/ref.float().square().mean().sqrt()),
                             'output_dtype': str(y.dtype)}
except Exception as exc:
    result['int8_convrot'] = {'error': repr(exc)}
pathlib.Path(__file__).with_name('kernel-probe.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(result['int8_convrot'])
