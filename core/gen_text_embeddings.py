"""Generate open scene descriptions and encode them with the frozen VLM.

Mean-pool intermediate text tokens into one embedding per image."""

import os
import json
import argparse
import numpy as np
import torch

# Config (match patch-token extraction)
MODEL_PATH = os.environ.get("QWEN_PATH",
                            "models/Qwen3.5-9B")  
LAYER_INDEX = -4
MAX_PIXELS = 1280 * 1280
GEN_PROMPT = ("Describe the weather conditions and the visible objects "
              "(vehicles, people) in this driving scene.")
MAX_NEW_TOKENS = 128

PREFIX_MAP = {
    "heavy_rain": "rain", "flooding": "flooding", "tornado": "tornado",
    "snow": "snow", "wildfire": "wildfire", "dust": "haze_sand_dust",
    "fog": "fog",
}


def infer_weather(fn):
    f = fn.lower()
    for p, w in PREFIX_MAP.items():
        if f.startswith(p):
            return w
    return "unknown"


import re


def _strip_thinking(text):
    """Remove reasoning tags and leading request restatements from generated text."""
    # remove explicit think tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"</?think>", "", text).strip()
    # Remove a generated request restatement.
    lines = text.split("\n")
    cleaned = []
    skipping = True
    for ln in lines:
        low = ln.strip().lower()
        if skipping and (low.startswith("the user wants")
                         or low.startswith("the user is asking")
                         or low.startswith("okay,") or low.startswith("sure,")
                         or low == ""):
            continue
        skipping = False
        cleaned.append(ln)
    out = "\n".join(cleaned).strip()
    return out if out else text   # never return empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xwod-root", default="data/xwod")
    ap.add_argument("--out-dir",
                    default="data/text_embeddings")
    ap.add_argument("--split", default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, only process this many images (smoke test)")
    ap.add_argument("--save-text", action="store_true", default=True,
                    help="also save the generated descriptions to JSON")
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText
    import cv2
    from PIL import Image

    img_dir = os.path.join(args.xwod_root,
                           "valid/images" if args.split == "val"
                           else f"{args.split}/images")
    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    if args.limit > 0:
        files = files[:args.limit]
    print(f"[{args.split}] processing {len(files)} images "
          f"(limit={args.limit or 'none'})")

    print("Loading Qwen (this can take a minute)...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH,
                                              max_pixels=MAX_PIXELS)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    device = next(model.parameters()).device
    # Set the padding token before generation.
    try:
        model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
    except Exception:
        pass

    os.makedirs(args.out_dir, exist_ok=True)
    embs = np.zeros((len(files), 4096), dtype=np.float16)
    valid = np.zeros(len(files), dtype=bool)
    weathers = []
    texts = {}

    for i, fn in enumerate(files):
        weathers.append(infer_weather(fn))
        try:
            img_bgr = cv2.imread(os.path.join(img_dir, fn))
            if img_bgr is None:
                raise RuntimeError("imread failed")
            img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

            # Step 1: generate description (image + prompt)
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": GEN_PROMPT}]}]
            # Remove a generated request restatement.
            
            try:
                inputs = processor.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                    enable_thinking=False).to(device)
            except TypeError:
                # Retry without enable_thinking for Transformers versions that do not accept it.
                inputs = processor.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False)
            new_tokens = gen[0][inputs["input_ids"].shape[1]:]
            desc = processor.decode(new_tokens,
                                    skip_special_tokens=True).strip()
            desc = _strip_thinking(desc)
            texts[fn] = desc

            # Step 2: encode the description with the same model
            # pure-text forward, take layer -4 hidden states, mean-pool
            tmsgs = [{"role": "user", "content":
                      [{"type": "text", "text": desc}]}]
            tin = processor.apply_chat_template(
                tmsgs, add_generation_prompt=False, tokenize=True,
                return_dict=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**tin, output_hidden_states=True)
            hs = out.hidden_states[LAYER_INDEX][0]   # [T, 4096]
            vec = hs.mean(dim=0).to(torch.float16).cpu().numpy()
            embs[i] = vec
            valid[i] = True
        except Exception as e:
            print(f"  [skip] {fn}: {e}")
            valid[i] = False

        if (i + 1) % 10 == 0:
            mem = (torch.cuda.max_memory_allocated() / 1024**3
                   if torch.cuda.is_available() else 0)
            print(f"  {i+1}/{len(files)}  (peak VRAM {mem:.1f} GB)")

    suffix = f"_limit{args.limit}" if args.limit > 0 else ""
    np.save(os.path.join(args.out_dir, f"{args.split}_text_emb{suffix}.npy"),
            embs)
    np.savez(os.path.join(args.out_dir, f"{args.split}_text_meta{suffix}.npz"),
             filenames=np.array(files), weathers=np.array(weathers),
             valid=valid)
    if args.save_text:
        with open(os.path.join(args.out_dir,
                               f"{args.split}_descriptions{suffix}.json"),
                  "w") as f:
            json.dump(texts, f, indent=2, ensure_ascii=False)

    print(f"\n  saved embeddings: {valid.sum()}/{len(files)} valid")
    print(f"  out dir: {args.out_dir}")
    # Print sample descriptions for inspection.
    print("\n  sample generated descriptions:")
    for fn in list(texts)[:3]:
        print(f"  [{infer_weather(fn)}] {fn}:")
        print(f"    {texts[fn][:200]}")


if __name__ == "__main__":
    main()
