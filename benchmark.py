import json
import pathlib
import sys
import time
import requests

ROOT = pathlib.Path(__file__).parent
URL = 'http://127.0.0.1:8190'
variant = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
api = json.loads((ROOT / (variant + '.api.json')).read_text(encoding='utf-8'))
workflow = json.loads((ROOT / (variant + '.workflow.json')).read_text(encoding='utf-8'))
tag = variant if seed is None else variant + '-seed' + str(seed)
if seed is not None:
    api['7']['inputs']['noise_seed'] = seed
    api['12']['inputs']['filename_prefix'] = tag
s = requests.Session()
s.post(URL + '/qwen_ab/reset_metrics', timeout=20).raise_for_status()
started = time.perf_counter()
r = s.post(URL + '/prompt', json={'prompt': api, 'extra_data': {'extra_pnginfo': {'workflow': workflow}}}, timeout=30)
if r.status_code != 200:
    (ROOT / (variant + '.validation-error.json')).write_text(r.text, encoding='utf-8')
    raise RuntimeError(r.text)
pid = r.json()['prompt_id']
print('RUN', tag, pid, flush=True)
samples = []
while True:
    metrics = s.get(URL + '/qwen_ab/metrics', timeout=30).json()
    metrics['elapsed_s'] = time.perf_counter() - started
    samples.append(metrics)
    history = s.get(URL + '/history/' + pid, timeout=30).json()
    if pid in history:
        break
    if len(samples) % 15 == 0:
        print('RUNNING', variant, round(metrics['elapsed_s']), 's, peak allocated GiB', round(metrics['peak_allocated_bytes']/1024**3, 3), flush=True)
    time.sleep(2)
result = {'variant': variant, 'prompt_id': pid, 'wall_seconds': time.perf_counter()-started,
          'metrics': metrics, 'sampled_device_used_peak_bytes': max(x['device_total_bytes']-x['device_free_bytes'] for x in samples),
          'history': history[pid], 'samples': samples}
(ROOT / (tag + '.result.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
print('FINISHED', tag, result['wall_seconds'], history[pid]['status'], metrics, flush=True)

