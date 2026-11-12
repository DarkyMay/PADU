"""Generate targeted scene descriptions and cache their frozen text embeddings."""
import os
import json
import argparse
import numpy as np
import torch
MODEL_PATH = os.environ.get('QWEN_PATH', 'models/Qwen3.5-9B')
LAYER_INDEX = -4
MAX_PIXELS = 1280 * 1280
GEN_PROMPT = 'Return compact JSON only. Consider people and road vehicles (cars, buses, trucks, motorcycles and bicycles). Describe the visible driving scene using keys weather, visibility (clear/reduced/severely_reduced), vehicle_count, person_count, small_distant_objects, largest_object_side (left/right/none), detection_difficulties. Counts should be estimated integers. For largest_object_side, use the traffic participant with the largest bounding-box area and compare its center with the image midline. Estimate from this image only; if uncertain use null. Do not assume detector outputs or ground truth.'
MAX_NEW_TOKENS = 128
PREFIX_MAP = {'heavy_rain': 'rain', 'flooding': 'flooding', 'tornado': 'tornado', 'snow': 'snow', 'wildfire': 'wildfire', 'dust': 'haze_sand_dust', 'fog': 'fog'}

def infer_weather(fn):
    f = fn.lower()
    for p, w in PREFIX_MAP.items():
        if f.startswith(p):
            return w
    return 'unknown'
import re

def _strip_thinking(text):
    """Remove reasoning tags and leading request restatements from generated text."""
    text = re.sub('<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub('</?think>', '', text).strip()
    lines = text.split('\n')
    cleaned = []
    skipping = True
    for ln in lines:
        low = ln.strip().lower()
        if skipping and (low.startswith('the user wants') or low.startswith('the user is asking') or low.startswith('okay,') or low.startswith('sure,') or (low == '')):
            continue
        skipping = False
        cleaned.append(ln)
    out = '\n'.join(cleaned).strip()
    return out if out else text

def main():
    from pathlib import Path
    import importlib.util
    engine_path = Path(__file__).resolve().parents[1] / 'targeted_text_batched.py'
    spec = importlib.util.spec_from_file_location('targeted_text_batched_engine', engine_path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.run_from_module(globals())
GENERATION_BATCH_SIZE = 4
BATCH_PLAN_SHA256 = 'e00226381ed0bce42c9418413d77292719151df00375f591b5777ada00394f87'

def _resume_initialize(args, files):
    from pathlib import Path
    import hashlib
    root = Path(__file__).resolve().parents[1]
    base = Path(args.out_dir).resolve()
    assert root in base.parents
    base.mkdir(parents=True, exist_ok=True)
    tag = args.split + ('_limit' + str(args.limit) if args.limit else '_full')
    paths = {k: base / (tag + suffix) for k, suffix in [('protocol', '_progress_protocol.json'), ('emb', '_incomplete_emb.npy'), ('flags', '_progress_flags.npz'), ('text', '_partial_descriptions.json')]}
    config = Path(MODEL_PATH) / 'config.json'
    protocol = {'files_sha256': hashlib.sha256(json.dumps(files).encode()).hexdigest(), 'n': len(files), 'model_path': MODEL_PATH, 'model_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(), 'prompt': GEN_PROMPT, 'max_new_tokens': MAX_NEW_TOKENS, 'max_pixels': MAX_PIXELS, 'layer': LAYER_INDEX, 'dtype': 'float16', 'do_sample': False, 'enable_thinking': False, 'generation_batch_size': GENERATION_BATCH_SIZE, 'batch_plan_sha256': BATCH_PLAN_SHA256}
    if paths['protocol'].exists():
        assert json.loads(paths['protocol'].read_text()) == protocol, 'Recovery protocol changed; inspect rather than mixing outputs'
    else:
        paths['protocol'].write_text(json.dumps(protocol, indent=2))
    if paths['emb'].exists():
        emb = np.load(paths['emb'], mmap_mode='r+')
        assert emb.shape == (len(files), 4096) and emb.dtype == np.float16
    else:
        emb = np.lib.format.open_memmap(paths['emb'], mode='w+', dtype=np.float16, shape=(len(files), 4096))
        emb[:] = 0
        emb.flush()
    flags = np.zeros(len(files), dtype=bool)
    texts = {}
    if paths['flags'].exists():
        with np.load(paths['flags'], allow_pickle=False) as saved:
            assert saved['filenames'].tolist() == files
            flags[:] = saved['valid']
        assert paths['text'].exists()
        texts = json.loads(paths['text'].read_text())
        assert all((f in texts for f, v in zip(files, flags) if v))
        flags &= np.isfinite(emb).all(1)
    print('RECOVERY_VALID_FRAMES', int(flags.sum()), 'of', len(files), flush=True)
    return (emb, flags, texts, paths)

def _save_generation_progress(emb, valid, texts, files, paths):
    emb.flush()
    tmp = paths['text'].with_name(paths['text'].name + '.tmp')
    tmp.write_text(json.dumps(texts, ensure_ascii=False), encoding='utf-8')
    tmp.replace(paths['text'])
    tmp = paths['flags'].with_name(paths['flags'].name + '.tmp')
    with tmp.open('wb') as f:
        np.savez(f, filenames=np.array(files), valid=valid)
    tmp.replace(paths['flags'])
if __name__ == '__main__':
    main()