"""Extract Qwen3.5-9B visual tokens for the manifest-defined BDD-IW splits."""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F


# Extraction settings

MODEL_PATH = "models/Qwen3.5-9B"  
BDD_ROOT = "data/BDD-IW"
MANIFEST = "data/bdd_understanding/splits.json"
OUTPUT_DIR = "data/bdd_patch_tokens"
LAYER_INDEX = -4
PROMPT = "Describe the weather degradation in this driving scene."
CHECKPOINT_EVERY = 100
MAX_PIXELS = 1280 * 1280  # identical to XWOD extraction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-h", type=int, default=12)
    ap.add_argument("--target-w", type=int, default=16)
    ap.add_argument("--split", default=None,
                    choices=["train", "val", "test"])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--bdd-root", default=BDD_ROOT)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    TARGET_H, TARGET_W = args.target_h, args.target_w
    K = TARGET_H * TARGET_W
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"  [WARN] MODEL_PATH not found: {MODEL_PATH}")
        print("         (XWOD script used this path; if your model is at a")
        print("          different location, e.g. Qwen3___5-9B, edit MODEL_PATH")
        print("          or pass it via env. Continuing; load may fail.)")

    # Resolve splits from manifest (test carved from physical train)
    with open(args.manifest) as f:
        man = json.load(f)
    # each split: {phys_split: 'train'|'val', files: [...]}
    splits_all = {s: (man[s]["phys_split"], man[s]["files"])
                  for s in ["train", "val", "test"]}
    splits = ({args.split: splits_all[args.split]}
              if args.split else splits_all)

    print("=" * 64)
    print("BDD-IW patch-token extraction (identical config to XWOD)")
    print("=" * 64)
    print(f"  Target pooled grid: {TARGET_H}x{TARGET_W} = {K} tokens/img")
    print(f"  MAX_PIXELS cap:     {MAX_PIXELS} (BDD is 720x1280, won't trigger)")
    for s, (phys, files) in splits.items():
        print(f"  split {s}: {len(files)} imgs (from physical {phys}/)")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, max_pixels=MAX_PIXELS)
    model.eval()
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    print(f"  Model loaded. image_pad_id={image_pad_id}, "
          f"VRAM={torch.cuda.memory_allocated()/1024**3:.2f} GB")

    def extract_patch_tokens(img_path):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": PROMPT},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        input_ids = inputs["input_ids"][0]
        img_mask = (input_ids == image_pad_id)
        n_img = int(img_mask.sum().item())

        grid_thw = inputs.get("image_grid_thw")
        if grid_thw is None or n_img == 0:
            del inputs
            torch.cuda.empty_cache()
            return None, n_img, None

        t, gh, gw = [int(v) for v in grid_thw[0].tolist()]
        h, w = gh // 2, gw // 2  # 2x2 spatial merge

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[LAYER_INDEX]
        img_tokens = hidden[0, img_mask, :]

        if h * w == n_img:
            grid = img_tokens.reshape(h, w, 4096)
        else:
            side = int(round(n_img ** 0.5))
            while side > 1 and n_img % side != 0:
                side -= 1
            h2 = max(1, side)
            w2 = n_img // h2
            grid = img_tokens[:h2 * w2].reshape(h2, w2, 4096)
            h, w = h2, w2

        grid = grid.permute(2, 0, 1).unsqueeze(0).float()
        pooled = F.adaptive_avg_pool2d(grid, (TARGET_H, TARGET_W))
        pooled = pooled.squeeze(0).permute(1, 2, 0).reshape(K, 4096)
        pooled_np = pooled.cpu().numpy().astype(np.float16)

        del out, hidden, inputs, img_tokens, grid, pooled
        torch.cuda.empty_cache()
        return pooled_np, n_img, (h, w)

    for split_name, (phys, files) in splits.items():
        print(f"\n{'=' * 64}\nProcessing split: {split_name}\n{'=' * 64}")
        img_dir = os.path.join(args.bdd_root, phys, "images")

        tok_path = os.path.join(args.output_dir, f"{split_name}_tokens.npy")
        meta_path = os.path.join(args.output_dir, f"{split_name}_meta.npz")
        ckpt_path = os.path.join(args.output_dir, f"{split_name}_ckpt.npz")

        if os.path.exists(tok_path) and os.path.exists(meta_path):
            existing = np.load(tok_path, mmap_mode="r")
            print(f"  Already done: {existing.shape}, skipping.")
            del existing
            continue

        all_files = sorted(files)  # manifest list for this split
        total = len(all_files)
        print(f"  Total images: {total}")

        start_idx = 0
        tokens_list, fnames_list = [], []
        weathers_list, nsrc_list, valid_list = [], [], []

        if os.path.exists(ckpt_path):
            ck = np.load(ckpt_path, allow_pickle=True)
            start_idx = int(ck["processed"])
            tokens_list = list(ck["tokens"])
            fnames_list = list(ck["filenames"])
            weathers_list = list(ck["weathers"])
            nsrc_list = list(ck["n_src_tokens"])
            valid_list = list(ck["valid"])
            print(f"  Resuming from checkpoint: {start_idx}/{total}")

        t_start = time.time()
        errors, n_invalid = [], 0

        for idx in range(start_idx, total):
            fname = all_files[idx]
            img_path = os.path.join(img_dir, fname)
            weather = "unknown"  # Weather labels are unavailable in these task inputs.

            try:
                pooled_np, n_src, hw = extract_patch_tokens(img_path)
                if pooled_np is None:
                    pooled_np = np.zeros((K, 4096), dtype=np.float16)
                    valid = False
                    n_invalid += 1
                else:
                    valid = True
            except Exception as e:
                pooled_np = np.zeros((K, 4096), dtype=np.float16)
                n_src, valid = 0, False
                n_invalid += 1
                errors.append((fname, str(e)))
                torch.cuda.empty_cache()

            tokens_list.append(pooled_np)
            fnames_list.append(fname)
            weathers_list.append(weather)
            nsrc_list.append(int(n_src))
            valid_list.append(bool(valid))

            processed = idx + 1
            if processed % 10 == 0 or processed == total:
                elapsed = time.time() - t_start
                done = processed - start_idx
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total - processed) / speed if speed > 0 else 0
                vram = torch.cuda.memory_allocated() / 1024**3
                print(f"  [{processed:5d}/{total}] {speed:.1f} img/s | "
                      f"ETA {eta/60:.1f}min | VRAM {vram:.1f}GB | "
                      f"invalid {n_invalid}", end="\r")

            if processed % CHECKPOINT_EVERY == 0:
                np.savez(
                    ckpt_path,
                    processed=processed,
                    tokens=np.array(tokens_list, dtype=np.float16),
                    filenames=np.array(fnames_list),
                    weathers=np.array(weathers_list),
                    n_src_tokens=np.array(nsrc_list),
                    valid=np.array(valid_list),
                )
        print()

        tokens_np = np.stack(tokens_list).astype(np.float16)
        np.save(tok_path, tokens_np)
        np.savez(meta_path,
                 filenames=np.array(fnames_list),
                 weathers=np.array(weathers_list),
                 n_src_tokens=np.array(nsrc_list),
                 valid=np.array(valid_list),
                 target_h=TARGET_H, target_w=TARGET_W, K=K)

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        elapsed = time.time() - t_start
        done = total - start_idx
        print(f"\n  Split [{split_name}] done:")
        print(f"    Tokens: {tokens_np.shape}  "
              f"({tokens_np.nbytes/1024**3:.2f} GB)")
        print(f"    Time: {elapsed/60:.1f} min ({done/elapsed:.1f} img/s)")
        print(f"    Invalid (no tokens / error): {n_invalid}")
        for fname, err in errors[:5]:
            print(f"      {fname}: {err}")

    print(f"\n{'=' * 64}\nEXTRACTION COMPLETE\n{'=' * 64}")
    total_gb = 0.0
    for split_name in splits:
        p = os.path.join(args.output_dir, f"{split_name}_tokens.npy")
        if os.path.exists(p):
            arr = np.load(p, mmap_mode="r")
            gb = arr.nbytes / 1024**3
            total_gb += gb
            print(f"  {split_name:6s}: {arr.shape}  ({gb:.2f} GB)")
            del arr
    print(f"  Total disk: {total_gb:.2f} GB")
    print(f"  Output dir: {args.output_dir}")
    if torch.cuda.is_available():
        print(f"  VRAM peak: "
              f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


if __name__ == "__main__":
    main()
