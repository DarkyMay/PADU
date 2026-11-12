"""Federated PaDU driver with checkpoint and client-partition exports.

This v2 implementation is selected by the corresponding experiment configs."""
import os
import sys
import json
import copy
import time
import math
import random
import argparse
import warnings
import contextlib
import io
import numpy as np
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')

@contextlib.contextmanager
def quiet():
    """Suppress repeated parameter-count messages when constructing client models."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield
WF_DIR_DEFAULT = 'core'

def import_model(wf_dir):
    sys.path.insert(0, wf_dir)
    from unified_model import UnifiedWeatherModel
    return UnifiedWeatherModel
_LOG = []
PROGRESS_ON = True

def W(s=''):
    s = str(s)
    _LOG.append(s)
    print(s, flush=True)

def flush_log(path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(_LOG))

def human_bytes(n):
    n = float(n)
    for u in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f'{n:.2f} {u}'
        n /= 1024
    return f'{n:.2f} PB'

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def _read_int(path):
    try:
        with open(path) as f:
            v = f.read().strip()
        return None if v in ('max', '') else int(v)
    except Exception:
        return None

def mem_limits():
    """Return memory limit, usage and source, preferring cgroup limits over host memory."""
    lim = _read_int('/sys/fs/cgroup/memory.max')
    use = _read_int('/sys/fs/cgroup/memory.current')
    if lim is not None:
        return (lim, use or 0, 'cgroup-v2')
    lim = _read_int('/sys/fs/cgroup/memory/memory.limit_in_bytes')
    use = _read_int('/sys/fs/cgroup/memory/memory.usage_in_bytes')
    if lim is not None and lim < 1 << 60:
        return (lim, use or 0, 'cgroup-v1')
    host = 0
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    host = int(line.split()[1]) * 1024
                    break
    except Exception:
        pass
    return (host, 0, 'host-meminfo(unbounded cgroup)')

def usable_ram_bytes(budget_gb=0.0):
    if budget_gb and budget_gb > 0:
        return int(budget_gb * (1 << 30))
    lim, use, _ = mem_limits()
    return max(0, int((lim - use) * 0.62))

class Prog:
    """Throttled progress bar on stderr; never enters the TXT report."""

    def __init__(self, total, label, enabled=True, width=28):
        self.total = max(1, int(total))
        self.label = label
        self.enabled = enabled and PROGRESS_ON
        self.width = width
        self.t0 = time.time()
        self.last = 0.0
        self.done = 0

    def update(self, n=1, extra=''):
        self.done += n
        if not self.enabled:
            return
        now = time.time()
        if now - self.last < 0.25 and self.done < self.total:
            return
        self.last = now
        frac = min(1.0, self.done / self.total)
        filled = int(self.width * frac)
        bar = '#' * filled + '-' * (self.width - filled)
        el = now - self.t0
        eta = el / frac - el if frac > 1e-09 else 0.0
        sys.stderr.write(f'\r    {self.label:<26} [{bar}] {self.done}/{self.total} {el:5.0f}s eta {eta:5.0f}s {extra}   ')
        sys.stderr.flush()

    def close(self):
        if self.enabled:
            sys.stderr.write('\r' + ' ' * (self.width + 90) + '\r')
            sys.stderr.flush()

class Split:
    """Load one dataset split with token storage on CPU, GPU or a memory map.

    Targets follow WeatherUnderstandingDataset: severity and detectability
    are three-class labels, density is [n_objects, n_small], count is the
    total, and side is left (0), right (1) or invalid (-1)."""

    def __init__(self, name, split, tok_root, label_dir, mode, device, verbose=True):
        self.name = name
        self.split = split
        self.mode = mode
        self.device = device
        tok_path = os.path.join(tok_root, f'{split}_tokens.npy')
        meta_path = os.path.join(tok_root, f'{split}_meta.npz')
        mm = np.load(tok_path, mmap_mode='r')
        meta = np.load(meta_path, allow_pickle=True)
        fnames = [str(x) for x in meta['filenames']]
        valid = np.asarray(meta['valid']).astype(bool)
        with open(os.path.join(label_dir, f'{split}_understanding_labels.json')) as f:
            u = json.load(f)
        with open(os.path.join(label_dir, f'{split}_detectability_labels.json')) as f:
            d = json.load(f)
        keep_pos, keep_f = ([], [])
        for i, fn in enumerate(fnames):
            if valid[i] and fn in u and (fn in d):
                keep_pos.append(i)
                keep_f.append(fn)
        keep_pos = np.asarray(keep_pos, dtype=np.int64)
        self.fnames = keep_f
        self.n = len(keep_f)
        self.weather = np.array([u[f].get('weather', 'unknown') for f in keep_f], dtype=object)
        self.severity = np.array([int(u[f]['severity']) for f in keep_f], dtype=np.int64)
        self.density = np.array([[float(u[f]['density']['n_objects']), float(u[f]['density']['n_small'])] for f in keep_f], dtype=np.float32)
        self.count = np.array([float(u[f]['vqa']['count_total']) for f in keep_f], dtype=np.float32)

        def side_of(f):
            s = u[f]['vqa']['largest_side']
            return 0 if s == 'left' else 1 if s == 'right' else -1
        self.side = np.array([side_of(f) for f in keep_f], dtype=np.int64)
        self.detect = np.array([int(d[f]['difficulty']) for f in keep_f], dtype=np.int64)
        self.det_score = np.array([float(d[f].get('score', np.nan)) for f in keep_f], dtype=np.float32)
        self.det_recall = np.array([float(d[f].get('recall', np.nan)) for f in keep_f], dtype=np.float32)
        nbytes = self.n * mm.shape[1] * mm.shape[2] * 2
        if verbose:
            W(f'  [{name}/{split}] usable {self.n} of {len(fnames)} | tokens {mm.shape[1]}x{mm.shape[2]} | {human_bytes(nbytes)}')
        self._mm = mm
        self._keep_pos = keep_pos
        self.nbytes = nbytes
        self.tokens = None
        if mode in ('gpu', 'ram'):
            K, D = (mm.shape[1], mm.shape[2])
            try:
                if mode == 'gpu':
                    buf = torch.empty((self.n, K, D), dtype=torch.float16, device=device)
                else:
                    buf = torch.empty((self.n, K, D), dtype=torch.float16)
                pr = Prog(self.n, f'load {name}/{split}')
                step = 512
                for i in range(0, self.n, step):
                    j = min(i + step, self.n)
                    blk = np.asarray(mm[keep_pos[i:j]], dtype=np.float16)
                    buf[i:j] = torch.from_numpy(blk).to(buf.device)
                    del blk
                    pr.update(j - i)
                pr.close()
                self.tokens = buf
            except (RuntimeError, MemoryError) as e:
                self.tokens = None
                if mode == 'gpu':
                    torch.cuda.empty_cache()
                W(f'    [{name}/{split}] {mode} load failed ({type(e).__name__}), falling back to memmap')
                self.mode = 'memmap'

    def get_tokens(self, pos):
        """pos: np.int64 array of positions within this Split."""
        if self.tokens is not None:
            idx = torch.from_numpy(pos)
            if self.tokens.is_cuda:
                idx = idx.to(self.tokens.device)
            t = self.tokens[idx]
            return t.to(self.device, non_blocking=True).float()
        arr = np.asarray(self._mm[self._keep_pos[pos]], dtype=np.float16)
        return torch.from_numpy(arr).to(self.device).float()

    def targets(self, pos):
        dev = self.device
        return {'severity': torch.from_numpy(self.severity[pos]).to(dev), 'density': torch.from_numpy(self.density[pos]).to(dev), 'count': torch.from_numpy(self.count[pos]).to(dev), 'side': torch.from_numpy(self.side[pos]).to(dev), 'detect': torch.from_numpy(self.detect[pos]).to(dev)}

def plan_placement(requested, sizes, device, budget_gb=0.0, reserve_gb=3.0):
    """Choose token storage per split, prioritizing train, validation, then test."""
    order = [s for s in ['train', 'val', 'test'] if s in sizes]
    if requested in ('gpu', 'ram', 'memmap'):
        return ({s: requested for s in order}, f'forced {requested}')
    plan = {s: 'memmap' for s in order}
    notes = []
    gpu_free = 0
    if device.type == 'cuda':
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            gpu_free = total - torch.cuda.memory_reserved(0)
            gpu_free = max(0, gpu_free - int(reserve_gb * (1 << 30)))
        except Exception:
            gpu_free = 0
    ram_free = usable_ram_bytes(budget_gb)
    notes.append(f'gpu_budget={human_bytes(gpu_free)} ram_budget={human_bytes(ram_free)}')
    for s in order:
        b = sizes[s]
        if b <= gpu_free:
            plan[s] = 'gpu'
            gpu_free -= b
        elif b <= ram_free:
            plan[s] = 'ram'
            ram_free -= b
    return (plan, '; '.join(notes))
CE = nn.CrossEntropyLoss()
CE_SIDE = nn.CrossEntropyLoss(ignore_index=-1)
SL1 = nn.SmoothL1Loss()

def batch_iter(idx, bs, shuffle, rng, drop_last=False):
    idx = np.asarray(idx)
    if shuffle:
        idx = idx.copy()
        rng.shuffle(idx)
    n = len(idx)
    last = n // bs * bs if drop_last else n
    for i in range(0, last, bs):
        yield idx[i:i + bs]

def compute_loss(out, tgt, lam, flip=False):
    sev = tgt['severity']
    det = tgt['detect']
    if flip:
        sev = 2 - sev
        det = 2 - det
    l_sev = CE(out['severity'], sev)
    l_det = CE(out['detect'], det)
    if (tgt['side'] >= 0).any():
        l_side = CE_SIDE(out['side'], tgt['side'])
    else:
        l_side = out['side'].sum() * 0.0
    l_dens = SL1(out['density'], torch.log1p(tgt['density']))
    l_cnt = SL1(out['count'], torch.log1p(tgt['count']))
    return l_sev + l_det + l_side + lam * (l_dens + l_cnt)

@torch.no_grad()
def evaluate(model, split, pos, device, bs=128):
    """pos: positions within split. Returns metric dict incl. balanced acc."""
    model.eval()
    n = 0
    sev_c = det_c = side_c = side_t = 0
    cnt_abs = 0.0
    dens_abs = np.zeros(2)
    sev_cm = np.zeros((3, 3), dtype=np.int64)
    det_cm = np.zeros((3, 3), dtype=np.int64)
    det_pred_all = np.zeros(len(pos), dtype=np.int64)
    ptr = 0
    rng = np.random.default_rng(0)
    for b in batch_iter(pos, bs, False, rng):
        tok = split.get_tokens(b)
        tgt = split.targets(b)
        out = model(tok)
        B = len(b)
        n += B
        sp = out['severity'].argmax(1).cpu().numpy()
        st = split.severity[b]
        sev_c += int((sp == st).sum())
        np.add.at(sev_cm, (st, sp), 1)
        dp = out['detect'].argmax(1).cpu().numpy()
        dt = split.detect[b]
        det_c += int((dp == dt).sum())
        np.add.at(det_cm, (dt, dp), 1)
        det_pred_all[ptr:ptr + B] = dp
        ptr += B
        cp = torch.expm1(out['count'].clamp(-10, 10)).clamp(min=0).cpu().numpy()
        cnt_abs += float(np.nan_to_num(np.abs(cp - split.count[b])).sum())
        de = torch.expm1(out['density'].clamp(-10, 10)).clamp(min=0).cpu().numpy()
        dens_abs += np.nan_to_num(np.abs(de - split.density[b])).sum(axis=0)
        sm = split.side[b]
        mask = sm >= 0
        if mask.sum():
            spd = out['side'].argmax(1).cpu().numpy()
            side_c += int((spd[mask] == sm[mask]).sum())
            side_t += int(mask.sum())

    def bal(cm):
        rec = []
        for i in range(cm.shape[0]):
            r = cm[i].sum()
            if r > 0:
                rec.append(cm[i, i] / r)
        return float(np.mean(rec)) if rec else 0.0
    m = {'n': int(n), 'severity_acc': sev_c / n, 'severity_bal': bal(sev_cm), 'detect_acc': det_c / n, 'detect_bal': bal(det_cm), 'side_acc': side_c / side_t if side_t else 0.0, 'count_mae': cnt_abs / n, 'nobj_mae': float(dens_abs[0] / n), 'nsmall_mae': float(dens_abs[1] / n), 'severity_cm': sev_cm.tolist(), 'detect_cm': det_cm.tolist()}
    m['combined'] = (m['severity_acc'] + m['detect_acc'] + m['side_acc']) / 3.0 - 0.02 * m['count_mae']
    m['_det_pred'] = det_pred_all
    return m

def fmt_metrics(m):
    return f'sevAcc {m['severity_acc']:.3f} (bal {m['severity_bal']:.3f})  detAcc {m['detect_acc']:.3f} (bal {m['detect_bal']:.3f})  sideAcc {m['side_acc']:.3f}  cntMAE {m['count_mae']:.2f}'

def strip_metrics(m):
    return {k: v for k, v in m.items() if not k.startswith('_')}

def local_train(model, split, pos, device, epochs, lr, wd, lam, bs, rng, mu=0.0, global_vec_params=None, flip=False, clip=5.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    nsteps = 0
    for _ in range(epochs):
        for b in batch_iter(pos, bs, True, rng):
            if len(b) < 2:
                continue
            tok = split.get_tokens(b)
            tgt = split.targets(b)
            out = model(tok)
            loss = compute_loss(out, tgt, lam, flip=flip)
            if mu > 0 and global_vec_params is not None:
                prox = 0.0
                for p, g in zip(model.parameters(), global_vec_params):
                    prox = prox + ((p - g) ** 2).sum()
                loss = loss + 0.5 * mu * prox
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            nsteps += 1
    return nsteps

def quantize_delta(delta, scheme):
    """Simulate a lossy uplink: the client sends a compressed update."""
    if scheme in ('none', None):
        return (delta, 4.0)
    if scheme == 'fp16':
        return (delta.half().float(), 2.0)
    if scheme == 'int8':
        s = delta.abs().max()
        if s < 1e-12:
            return (delta, 1.0)
        q = torch.clamp(torch.round(delta / s * 127.0), -127, 127)
        return (q * s / 127.0, 1.0)
    raise ValueError(scheme)

def _flatten(state, keys):
    return torch.cat([state[k].reshape(-1).float() for k in keys])

def _unflatten(vec, ref, keys):
    out, i = ({}, 0)
    for k in keys:
        n = ref[k].numel()
        out[k] = vec[i:i + n].view_as(ref[k]).clone()
        i += n
    return out

def aggregate(states, weights, rule='fedavg', trim=1, n_byz=1):
    """states: list of state_dicts (cpu float). weights: np array, sums to 1."""
    keys = list(states[0].keys())
    M = len(states)
    for st in states:
        for k in keys:
            if not torch.isfinite(st[k]).all():
                st[k] = torch.nan_to_num(st[k], nan=0.0, posinf=0.0, neginf=0.0)
    if rule == 'fedavg':
        out = {}
        w = torch.tensor(weights, dtype=torch.float32)
        for k in keys:
            stack = torch.stack([s[k].float() for s in states])
            shape = [M] + [1] * (stack.dim() - 1)
            out[k] = (stack * w.view(*shape)).sum(0)
        return out
    if rule == 'median':
        out = {}
        for k in keys:
            stack = torch.stack([s[k].float() for s in states])
            out[k] = stack.median(dim=0).values
        return out
    if rule == 'trimmed_mean':
        t = min(trim, max(0, (M - 1) // 2))
        out = {}
        for k in keys:
            stack = torch.stack([s[k].float() for s in states])
            srt, _ = torch.sort(stack, dim=0)
            if t > 0:
                srt = srt[t:M - t]
            out[k] = srt.mean(dim=0)
        return out
    if rule in ('krum', 'multi_krum'):
        vecs = torch.stack([_flatten(s, keys) for s in states])
        d = torch.cdist(vecs.unsqueeze(0), vecs.unsqueeze(0)).squeeze(0) ** 2
        k = max(1, M - n_byz - 2)
        scores = []
        for i in range(M):
            di = torch.cat([d[i, :i], d[i, i + 1:]])
            scores.append(torch.sort(di).values[:k].sum().item())
        order = np.argsort(scores)
        if rule == 'krum':
            sel = order[:1]
        else:
            sel = order[:max(1, M - n_byz)]
        w = np.asarray([weights[i] for i in sel], dtype=np.float64)
        w = w / w.sum()
        out = {}
        for kk in keys:
            stack = torch.stack([states[i][kk].float() for i in sel])
            shape = [len(sel)] + [1] * (stack.dim() - 1)
            out[kk] = (stack * torch.tensor(w, dtype=torch.float32).view(*shape)).sum(0)
        return out
    raise ValueError(rule)

def new_model(ModelCls, device, cfg, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    with quiet():
        m = ModelCls(d_model=cfg['d_model'], n_layers=cfg['n_layers'], n_heads=cfg['n_heads'], dropout=cfg['dropout'])
    return m.to(device)

def count_params(model):
    enc = sum((p.numel() for n, p in model.named_parameters() if n.startswith('encoder.')))
    head = sum((p.numel() for n, p in model.named_parameters() if n.startswith('head_')))
    return (enc, head, enc + head)

def run_centralized(ModelCls, tr, va, te, device, cfg, tag='centralized', epochs=None, verbose_every=10):
    import re
    from pathlib import Path
    _saved = Path(os.environ['PADU_RUN_OUT']) / 'checkpoints' / f'seed{cfg['seed']}_{re.sub('[^a-zA-Z0-9_-]', '_', tag)}.pt'
    if _saved.exists():
        _bundle = torch.load(_saved, map_location=device, weights_only=False)
        assert _bundle['config'] == cfg and _bundle['best_state'] is not None
        model = new_model(ModelCls, device, cfg, seed=cfg['seed'])
        model.load_state_dict(_bundle['best_state'])
        mt = evaluate(model, te, np.arange(te.n), device, cfg['eval_bs'])
        _saved.with_suffix('.sha256').write_text(hashlib.sha256(_saved.read_bytes()).hexdigest() + '  ' + _saved.name + '\n')
        print('RESUMED_COMPLETED_CENTRAL_CHECKPOINT', str(_saved), flush=True)
        return (model, mt)
    epochs = epochs or cfg['cen_epochs']
    set_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    model = new_model(ModelCls, device, cfg, seed=cfg['seed'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-06)
    all_tr = np.arange(tr.n)
    all_va = np.arange(va.n)
    all_te = np.arange(te.n)
    best, bad, best_state = (-1000000000.0, 0, None)
    t0 = time.time()
    pr = Prog(epochs, f'{tag}')
    for ep in range(epochs):
        model.train()
        for b in batch_iter(all_tr, cfg['bs'], True, rng, drop_last=True):
            tok = tr.get_tokens(b)
            tgt = tr.targets(b)
            loss = compute_loss(model(tok), tgt, cfg['lam'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        m = evaluate(model, va, all_va, device, cfg['eval_bs'])
        pr.update(1, f'val comb {m['combined']:.4f}')
        if (ep + 1) % verbose_every == 0:
            W(f'    [{tag}] ep {ep + 1:3d}/{epochs}  val {fmt_metrics(m)}')
        if m['combined'] > best + 0.0001:
            best, bad = (m['combined'], 0)
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg['patience']:
                pr.close()
                W(f'    [{tag}] early stop @ ep {ep + 1}')
                break
    pr.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    mt = evaluate(model, te, all_te, device, cfg['eval_bs'])
    W(f'    [{tag}] TEST {fmt_metrics(mt)}   ({time.time() - t0:.0f}s)')
    _revision_save(locals(), 'run_centralized')
    return (model, mt)

def run_federated(ModelCls, tr, va, te, device, cfg, clients, algo='fedavg', agg_rule='fedavg', byz_ids=(), attack='none', tag='fedavg', personalized_eval=None, val_eval=None, quant='none', rounds=None):
    """
    clients: list of dicts {name, idx (np array of train positions)}
    algo   : fedavg | fedprox | fedper | local
    attack : none | label_flip | sign_flip | gauss
    personalized_eval: optional dict client_name -> np array of test positions
    """
    set_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    R = rounds or cfg['rounds']
    E = cfg['local_epochs']
    all_va = np.arange(va.n)
    all_te = np.arange(te.n)
    gmodel = new_model(ModelCls, device, cfg, seed=cfg['seed'])
    enc_p, head_p, tot_p = count_params(gmodel)
    shared_keys = [k for k in gmodel.state_dict().keys() if algo != 'fedper' or k.startswith('encoder.')]
    _, bpp = quantize_delta(torch.zeros(1), quant)
    n_shared = sum((gmodel.state_dict()[k].numel() for k in shared_keys))
    bytes_per_up = int(bpp * n_shared)
    private = {c['name']: None for c in clients}
    curves = []
    best, best_state, bad = (-1000000000.0, None, 0)
    best_round = 0
    best_private = None
    cum_uplink = 0
    t0 = time.time()
    pr = Prog(R, f'{tag}')
    for rd in range(R):
        m_sel = max(1, int(round(cfg['frac'] * len(clients))))
        sel = list(rng.choice(len(clients), size=m_sel, replace=False)) if cfg['frac'] < 1.0 else list(range(len(clients)))
        gstate = copy.deepcopy(gmodel.state_dict())
        gparams = [p.detach().clone() for p in gmodel.parameters()] if algo == 'fedprox' else None
        states, weights, names = ([], [], [])
        lr_rd = cfg['lr'] * 0.5 * (1 + math.cos(math.pi * rd / max(R - 1, 1)))
        lr_rd = max(lr_rd, 1e-06)
        for ci in sel:
            c = clients[ci]
            lm = new_model(ModelCls, device, cfg)
            if algo == 'local':
                lm.load_state_dict(private[c['name']] or gstate)
            else:
                st = copy.deepcopy(gstate)
                if algo == 'fedper' and private[c['name']] is not None:
                    for k, v in private[c['name']].items():
                        st[k] = v
                lm.load_state_dict(st)
            flip = attack == 'label_flip' and ci in byz_ids
            local_train(lm, tr, c['idx'], device, E, lr_rd, cfg['wd'], cfg['lam'], cfg['bs'], rng, mu=cfg['mu'] if algo == 'fedprox' else 0.0, global_vec_params=gparams, flip=flip)
            lstate = {k: v.detach().cpu().float() for k, v in lm.state_dict().items()}
            if algo in ('fedper', 'local'):
                private[c['name']] = {k: v.clone() for k, v in lstate.items() if algo == 'local' or k.startswith('head_')}
            if quant != 'none':
                g_cpu = {k: v.detach().cpu().float() for k, v in gstate.items()}
                for k in shared_keys:
                    dq, _ = quantize_delta(lstate[k] - g_cpu[k], quant)
                    lstate[k] = g_cpu[k] + dq
            if ci in byz_ids and attack in ('sign_flip', 'gauss'):
                g_cpu = {k: v.detach().cpu().float() for k, v in gstate.items()}
                for k in lstate:
                    delta = lstate[k] - g_cpu[k]
                    if attack == 'sign_flip':
                        lstate[k] = g_cpu[k] - cfg['byz_scale'] * delta
                    else:
                        rms = float(delta.pow(2).mean().sqrt()) + 1e-08
                        noise = torch.randn_like(delta) * rms * cfg['byz_scale']
                        lstate[k] = g_cpu[k] + noise
            states.append(lstate)
            weights.append(len(c['idx']))
            names.append(c['name'])
            del lm
        cum_uplink += bytes_per_up * len(states)
        if algo != 'local':
            w = np.asarray(weights, dtype=np.float64)
            w = w / w.sum()
            g_ref = {k: gstate[k].detach().cpu().float() for k in shared_keys}
            deltas = [{k: s[k] - g_ref[k] for k in shared_keys} for s in states]
            agg = aggregate(deltas, w, rule=agg_rule, trim=cfg['trim'], n_byz=max(1, len(byz_ids)))
            gs = gmodel.state_dict()
            for k in shared_keys:
                upd = torch.nan_to_num(g_ref[k] + agg[k], nan=0.0, posinf=0.0, neginf=0.0)
                gs[k] = upd.to(device)
            gmodel.load_state_dict(gs)
        if algo == 'local':
            mv = {'combined': -1000000000.0}
        elif algo == 'fedper':
            gs_now = {k: v.detach().cpu().clone() for k, v in gmodel.state_dict().items()}
            pv = personalized_test(ModelCls, private, clients, va, device, cfg, val_eval, base_state=gmodel.state_dict())
            mv = pv['pooled']
            if mv['combined'] > best + 0.0001:
                best, bad = (mv['combined'], 0)
                best_round = rd + 1
                best_state = gs_now
                best_private = {k: {kk: vv.clone() for kk, vv in v.items()} if v else None for k, v in private.items()}
            else:
                bad += 1
        else:
            mv = evaluate(gmodel, va, all_va, device, cfg['eval_bs'])
            if mv['combined'] > best + 0.0001:
                best, bad = (mv['combined'], 0)
                best_round = rd + 1
                best_state = copy.deepcopy(gmodel.state_dict())
            else:
                bad += 1
        curves.append({'round': rd + 1, 'uplink_bytes': cum_uplink, 'val_combined': mv.get('combined'), 'val_detect_acc': mv.get('detect_acc'), 'val_severity_acc': mv.get('severity_acc'), 'val_side_acc': mv.get('side_acc')})
        pr.update(1, f'val comb {mv['combined']:.4f}' if algo != 'local' else 'local')
        if (rd + 1) % cfg['log_every'] == 0 and algo != 'local':
            W(f'    [{tag}] round {rd + 1:3d}/{R}  uplink {human_bytes(cum_uplink)}  val {fmt_metrics(mv)}')
        if algo != 'local' and bad >= cfg['fed_patience']:
            pr.close()
            W(f'    [{tag}] early stop @ round {rd + 1}')
            break
    pr.close()
    res = {'tag': tag, 'algo': algo, 'agg_rule': agg_rule, 'attack': attack, 'n_clients': len(clients), 'byz': [clients[i]['name'] for i in byz_ids], 'rounds_run': len(curves), 'uplink_bytes_per_client_round': bytes_per_up, 'uplink_bytes_total': cum_uplink, 'params_total': tot_p, 'params_encoder': enc_p, 'params_head': head_p, 'curves': curves, 'seconds': time.time() - t0, 'best_round': best_round, 'quant': quant}
    if algo == 'local':
        res['global'] = None
        pm = personalized_test(ModelCls, private, clients, te, device, cfg, personalized_eval)
        res['personalized'] = pm
        W(f'    [{tag}] personalized TEST {fmt_metrics(pm['pooled'])}  ({res['seconds']:.0f}s)')
    elif algo == 'fedper':
        if best_state is not None:
            gmodel.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        use_priv = best_private if best_private is not None else private
        res['global'] = None
        res['global_note'] = 'FedPer aggregates only the encoder; it has no global head, so no global row is reported.'
        pm = personalized_test(ModelCls, use_priv, clients, te, device, cfg, personalized_eval, base_state=gmodel.state_dict())
        res['personalized'] = pm
        W(f'    [{tag}] personalized TEST {fmt_metrics(pm['pooled'])}  ({res['seconds']:.0f}s)')
    else:
        final_state = copy.deepcopy(gmodel.state_dict())
        mt_fin = evaluate(gmodel, te, all_te, device, cfg['eval_bs'])
        res['global_final'] = strip_metrics(mt_fin)
        if best_state is not None:
            gmodel.load_state_dict(best_state)
        mt = evaluate(gmodel, te, all_te, device, cfg['eval_bs'])
        res['global'] = strip_metrics(mt)
        res['selection_fell_back'] = bool(best_round <= 1)
        note = '  <-- best-val selection fell back to the init model' if best_round <= 1 else ''
        W(f'    [{tag}] global TEST (best-val, round {best_round}) {fmt_metrics(mt)}{note}')
        W(f'    [{tag}] global TEST (final round)   {fmt_metrics(mt_fin)}  ({res['seconds']:.0f}s)')
        gmodel.load_state_dict(final_state)
    _revision_save(locals(), 'run_federated')
    return (res, gmodel)

def personalized_test(ModelCls, private, clients, te, device, cfg, eval_map, base_state=None):
    """Each client evaluated on the test slice matching its own domain."""
    per, pooled_cm_s, pooled_cm_d = ({}, np.zeros((3, 3), np.int64), np.zeros((3, 3), np.int64))
    tot = sev_c = det_c = side_c = side_t = 0
    cnt_abs = 0.0
    for c in clients:
        name = c['name']
        pos = eval_map.get(name) if eval_map else None
        if pos is None or len(pos) == 0 or private[name] is None:
            continue
        lm = new_model(ModelCls, device, cfg)
        st = copy.deepcopy(base_state) if base_state is not None else lm.state_dict()
        for k, v in private[name].items():
            st[k] = v.to(device)
        lm.load_state_dict(st)
        m = evaluate(lm, te, pos, device, cfg['eval_bs'])
        per[name] = strip_metrics(m)
        n = m['n']
        tot += n
        sev_c += m['severity_acc'] * n
        det_c += m['detect_acc'] * n
        cnt_abs += m['count_mae'] * n
        pooled_cm_s += np.asarray(m['severity_cm'])
        pooled_cm_d += np.asarray(m['detect_cm'])
        side_c += m['side_acc'] * n
        side_t += n
        del lm

    def bal(cm):
        rec = [cm[i, i] / cm[i].sum() for i in range(3) if cm[i].sum() > 0]
        return float(np.mean(rec)) if rec else 0.0
    pooled = {'n': tot, 'severity_acc': sev_c / max(tot, 1), 'severity_bal': bal(pooled_cm_s), 'detect_acc': det_c / max(tot, 1), 'detect_bal': bal(pooled_cm_d), 'side_acc': side_c / max(side_t, 1), 'count_mae': cnt_abs / max(tot, 1)}
    pooled['combined'] = (pooled['severity_acc'] + pooled['detect_acc'] + pooled['side_acc']) / 3.0 - 0.02 * pooled['count_mae']
    return {'pooled': pooled, 'per_client': per}

def build_partition(tr, va, te, mode, k, seed, min_n=8):
    """Build client partitions and per-client validation and test indices.

    Supported partitions are weather (XWOD), object density with training-set
    thresholds, detectability level, and random shards."""
    if mode == 'weather':
        names = [w for w in sorted(set(tr.weather.tolist())) if int((tr.weather == w).sum()) >= min_n]

        def assign(sp):
            return sp.weather.astype(object)
    elif mode == 'density':
        qs = np.quantile(tr.density[:, 0], np.linspace(0, 1, k + 1))[1:-1]
        names = [f'dens{i}' for i in range(k)]
        lo = [float(tr.density[:, 0].min())] + list(qs)
        hi = list(qs) + [float(tr.density[:, 0].max())]
        W(f'  density bands (objects/image): ' + ', '.join((f'{names[i]}=[{lo[i]:.1f},{hi[i]:.1f}]' for i in range(k))))

        def assign(sp):
            b = np.searchsorted(qs, sp.density[:, 0], side='right')
            return np.array([names[min(int(x), k - 1)] for x in b], dtype=object)
    elif mode == 'detect':
        names = ['easy', 'medium', 'hard']

        def assign(sp):
            return np.array([names[int(d)] for d in sp.detect], dtype=object)
    elif mode == 'random':
        names = [f'shard{i}' for i in range(k)]

        def assign(sp):
            r = np.random.default_rng(seed + len(sp.fnames))
            return np.array([names[i] for i in r.integers(0, k, size=sp.n)], dtype=object)
    else:
        raise ValueError(mode)
    a_tr, a_va, a_te = (assign(tr), assign(va), assign(te))
    clients = []
    for nm in names:
        idx = np.where(a_tr == nm)[0]
        if len(idx) >= min_n:
            clients.append({'name': str(nm), 'idx': idx})
    clients.sort(key=lambda c: -len(c['idx']))
    va_map = {c['name']: np.where(a_va == c['name'])[0] for c in clients}
    te_map = {c['name']: np.where(a_te == c['name'])[0] for c in clients}
    return (clients, va_map, te_map)

def weather_clients(split):
    out = []
    for w in sorted(set(split.weather.tolist())):
        idx = np.where(split.weather == w)[0]
        if len(idx) >= 8:
            out.append({'name': str(w), 'idx': idx})
    out.sort(key=lambda c: -len(c['idx']))
    return out

def shard_clients(split, k, seed, prefix='shard'):
    rng = np.random.default_rng(seed)
    idx = np.arange(split.n)
    rng.shuffle(idx)
    parts = np.array_split(idx, k)
    return [{'name': f'{prefix}{i}', 'idx': p} for i, p in enumerate(parts)]

def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

@torch.no_grad()
def feature_heterogeneity(split, clients, device, bs=128):
    """Compare client feature centroids using cosine similarity.

    Within-weather severity tertiles have balanced class frequencies by
    construction, so feature centroids provide an additional measure of heterogeneity."""
    cents = []
    names = []
    for c in clients:
        acc = torch.zeros(4096, dtype=torch.float64, device=device)
        n = 0
        rng = np.random.default_rng(0)
        for b in batch_iter(c['idx'], bs, False, rng):
            tok = split.get_tokens(b)
            acc += tok.mean(dim=1).sum(dim=0).double()
            n += len(b)
        cents.append((acc / max(n, 1)).float())
        names.append(c['name'])
    Cm = torch.stack(cents)
    Cn = Cm / Cm.norm(dim=1, keepdim=True).clamp(min=1e-09)
    S = (Cn @ Cn.t()).cpu().numpy()
    W('\n  Feature-level heterogeneity: cosine similarity between the per-client mean')
    W('  frozen-VLM token embeddings (1.000 = identical feature distribution)')
    W('  ' + ' ' * 16 + ''.join((f'{n[:7]:>9}' for n in names)))
    for i, n in enumerate(names):
        W(f'  {n:<16}' + ''.join((f'{S[i, j]:>9.3f}' for j in range(len(names)))))
    off = S[~np.eye(len(names), dtype=bool)]
    W(f'  mean off-diagonal cosine = {off.mean():.3f}  (min {off.min():.3f}, max {off.max():.3f})')
    W('  Lower off-diagonal similarity = more feature-level non-IID.')
    return {'clients': names, 'cosine': S.tolist(), 'mean_off_diag': float(off.mean()), 'min_off_diag': float(off.min())}

def noniid_report(split, clients):
    W('\n  Client partition and non-IID statistics (train split)')
    W('  sev / det columns are the percentage split over the three ordinal levels within that client')
    W(f'  {'client':<18}{'n':>6}{'share':>8}{'severity %':>18}{'detect %':>18}{'objs':>7}{'JS(det)':>9}{'JS(sev)':>9}')
    gdet = np.bincount(split.detect, minlength=3)
    gsev = np.bincount(split.severity, minlength=3)
    rows = []
    for c in clients:
        i = c['idx']
        ds = np.bincount(split.detect[i], minlength=3)
        ss = np.bincount(split.severity[i], minlength=3)
        objs = float(split.density[i, 0].mean())
        jd = js_divergence(ds, gdet)
        jsv = js_divergence(ss, gsev)

        def pct(a):
            t = max(a.sum(), 1)
            return '/'.join((f'{v / t * 100:.0f}' for v in a))
        W(f'  {c['name']:<18}{len(i):>6}{len(i) / split.n * 100:>7.1f}%{pct(ss):>18}{pct(ds):>18}{objs:>7.1f}{jd:>9.4f}{jsv:>9.4f}')
        rows.append({'client': c['name'], 'n': int(len(i)), 'severity_dist': ss.tolist(), 'detect_dist': ds.tolist(), 'mean_objects': objs, 'js_detect': jd, 'js_severity': jsv})
    sizes = [len(c['idx']) for c in clients]
    W(f'  size imbalance  max/min = {max(sizes) / max(min(sizes), 1):.1f}x   mean JS(detect) = {np.mean([r['js_detect'] for r in rows]):.4f}')
    return rows

def exp0_sanity(ModelCls, tr, va, te, device, cfg, results):
    W('\n' + '=' * 78)
    W('EXP-0  Centralized reference comparison')
    W('=' * 78)
    _, mt = run_centralized(ModelCls, tr, va, te, device, cfg, tag='centralized')
    results['exp0_centralized'] = strip_metrics(mt)
    ref = {'severity_acc': 0.624, 'detect_acc': 0.517, 'side_acc': 0.802, 'count_mae': 1.362}
    W('\n  vs. paper (mean over 3 seeds):')
    W(f'  {'metric':<16}{'here':>10}{'paper':>10}{'delta':>10}')
    for k, v in ref.items():
        W(f'  {k:<16}{mt[k]:>10.3f}{v:>10.3f}{mt[k] - v:>+10.3f}')
    dmax = max((abs(mt[k] - v) / (1.0 if k.endswith('_acc') else abs(v) / 0.6) for k, v in ref.items()))
    if dmax <= 0.03:
        W(f'  VERDICT: matches the paper (max |delta| = {dmax:.3f}).')
    else:
        W(f'  VERDICT: max |delta| = {dmax:.3f}. If this run used --quick, that is just\n           undertraining (3 epochs, cosine T_max=3). On a full run it must be <=0.03,\n           otherwise the data layer is wrong and nothing below is trustworthy.')
    return mt

def expA_federation(ModelCls, tr, va, te, device, cfg, results, cen_test):
    W('\n' + '=' * 78)
    W('EXP-A  Federated PaDU over non-IID weather clients')
    W('=' * 78)
    clients, va_map, te_map = build_partition(tr, va, te, cfg['client_split'], cfg['n_shards'], cfg['seed'])
    results['client_split'] = cfg['client_split']
    results['clients'] = noniid_report(tr, clients)
    results['feature_heterogeneity'] = feature_heterogeneity(tr, clients, device)
    W('\n  NOTE: severity JS is 0.0000 for every client by construction — the XWOD severity\n  tertiles are fit within each weather type, so each client is 33/33/33. The real\n  heterogeneity is in detectability, object density, client size, and the features.')
    runs = {}
    W('\n  -- Local-only (no communication) --')
    runs['local'], _ = run_federated(ModelCls, tr, va, te, device, cfg, clients, algo='local', tag='local-only', personalized_eval=te_map)
    W('\n  -- FedAvg --')
    runs['fedavg'], _ = run_federated(ModelCls, tr, va, te, device, cfg, clients, algo='fedavg', tag='FedAvg')
    W('\n  -- FedProx (mu=%.3f) --' % cfg['mu'])
    runs['fedprox'], _ = run_federated(ModelCls, tr, va, te, device, cfg, clients, algo='fedprox', tag='FedProx')
    W('\n  -- FedPer (shared encoder, personalized heads) --')
    runs['fedper'], _ = run_federated(ModelCls, tr, va, te, device, cfg, clients, algo='fedper', tag='FedPer', personalized_eval=te_map, val_eval=va_map)
    W('\n  -- FedAvg uplink compression (accuracy vs. bytes) --')
    for q in ['fp16', 'int8']:
        runs[f'fedavg_{q}'], _ = run_federated(ModelCls, tr, va, te, device, cfg, clients, algo='fedavg', tag=f'FedAvg+{q}', quant=q)
    results['expA'] = runs
    W('\n  Summary (XWOD global test set)')
    W(f'  {'method':<26}{'sevAcc':>9}{'detAcc':>9}{'detBal':>9}{'sideAcc':>9}{'cntMAE':>9}')
    W(f'  {'Centralized (upper bnd)':<26}{cen_test['severity_acc']:>9.3f}{cen_test['detect_acc']:>9.3f}{cen_test['detect_bal']:>9.3f}{cen_test['side_acc']:>9.3f}{cen_test['count_mae']:>9.2f}')
    for key, label in [('fedavg', 'FedAvg'), ('fedprox', 'FedProx'), ('fedavg_fp16', 'FedAvg + fp16 uplink'), ('fedavg_int8', 'FedAvg + int8 uplink')]:
        g = runs.get(key, {}).get('global')
        if not g:
            continue
        W(f'  {label:<26}{g['severity_acc']:>9.3f}{g['detect_acc']:>9.3f}{g['detect_bal']:>9.3f}{g['side_acc']:>9.3f}{g['count_mae']:>9.2f}')
    W('  (FedPer has no global head, so it appears only in the table below.)')
    W("\n  Personalized evaluation (each client on its own weather's test slice)")
    W(f'  {'method':<26}{'sevAcc':>9}{'detAcc':>9}{'detBal':>9}{'sideAcc':>9}{'cntMAE':>9}')
    for key, label in [('local', 'Local-only'), ('fedper', 'FedPer')]:
        p = runs[key].get('personalized', {}).get('pooled')
        if p:
            W(f'  {label:<26}{p['severity_acc']:>9.3f}{p['detect_acc']:>9.3f}{p['detect_bal']:>9.3f}{p['side_acc']:>9.3f}{p['count_mae']:>9.2f}')
    return runs

def expB_communication(ModelCls, device, cfg, results, runs):
    W('\n' + '=' * 78)
    W('EXP-B  Communication accounting')
    W('=' * 78)
    m = new_model(ModelCls, device, cfg)
    enc, head, tot = count_params(m)
    del m
    VLM_PARAMS = 9000000000.0
    TOKENS_PER_IMG = 192 * 4096 * 2
    IMG_BYTES = 200 * 1024
    fa = runs.get('fedavg', {})
    rounds = fa.get('rounds_run', cfg['rounds'])
    n_cl = fa.get('n_clients', 7)
    per_round_client = tot * 4
    fedavg_total = fa.get('uplink_bytes_total', per_round_client * n_cl * rounds)
    fedper_total = runs.get('fedper', {}).get('uplink_bytes_total', enc * 4 * n_cl * rounds)
    RN = rounds
    rows = [('PaDU decoder, full (FedAvg)', tot, per_round_client), ('PaDU decoder, fp16 uplink', tot, tot * 2), ('PaDU decoder, int8 uplink', tot, tot * 1), ('PaDU encoder only (FedPer, fp32)', enc, enc * 4), ('Fine-tuning the frozen VLM (hypothetical)', int(VLM_PARAMS), int(VLM_PARAMS * 4))]
    W(f'\n  {'transmitted object':<42}{'params':>13}{'per client/round':>19}{f'x{n_cl} clients x{RN} rd':>22}')
    for name, prm, pr in rows:
        W(f'  {name:<42}{prm / 1000000.0:>11.2f}M{human_bytes(pr):>19}{human_bytes(pr * n_cl * RN):>22}')
    W(f'\n  Uplink reduction vs. VLM fine-tuning : {VLM_PARAMS / tot:>.0f}x (fp32), {VLM_PARAMS * 2 / tot:>.0f}x (int8)')
    W(f'  FedPer sends only the encoder: {(1 - enc / tot) * 100:.1f}% saved per round')
    W('  Measured per-run totals (different early-stop rounds, not directly comparable):')
    for key, label in [('fedavg', 'FedAvg'), ('fedper', 'FedPer'), ('fedavg_fp16', 'FedAvg+fp16'), ('fedavg_int8', 'FedAvg+int8')]:
        r = runs.get(key)
        if r:
            W(f'    {label:<14}{r['rounds_run']:>4} rounds{human_bytes(r['uplink_bytes_total']):>14}')
    W('\n  Centralized alternative (what federation avoids uploading):')
    for name, n_img in [('XWOD train images (raw jpg)', 5944), ('XWOD train cached tokens (fp16)', 5944)]:
        b = n_img * (IMG_BYTES if 'jpg' in name else TOKENS_PER_IMG)
        W(f'    {name:<40}{human_bytes(b):>14}')
    W(f"    one image's cached tokens{'':<15}{human_bytes(TOKENS_PER_IMG):>14}   ({TOKENS_PER_IMG / per_round_client:.2f} model-updates worth)")
    W('\n  Uplink compression: accuracy vs. bytes (same rounds, same clients)')
    W(f'  {'uplink scheme':<24}{'bytes/param':>13}{'total uplink':>16}{'detAcc':>9}{'sevAcc':>9}{'sideAcc':>9}{'cntMAE':>9}')
    for key, label in [('fedavg', 'fp32 (uncompressed)'), ('fedavg_fp16', 'fp16'), ('fedavg_int8', 'int8')]:
        r = runs.get(key)
        if not r or not r.get('global'):
            continue
        g = r['global']
        bpp = r['uplink_bytes_per_client_round'] / max(runs['fedavg']['uplink_bytes_per_client_round'] / 4.0, 1)
        W(f'  {label:<24}{bpp:>13.1f}{human_bytes(r['uplink_bytes_total']):>16}{g['detect_acc']:>9.3f}{g['severity_acc']:>9.3f}{g['side_acc']:>9.3f}{g['count_mae']:>9.2f}')
    W('\n  Uplink needed to first reach a given fraction of the centralized val score')
    cen_ref = results.get('exp0_centralized', {})
    W(f'  {'scheme':<16}{'50%':>14}{'80%':>14}{'90%':>14}{'95%':>14}')
    for key, label in [('fedavg', 'fp32'), ('fedavg_fp16', 'fp16'), ('fedavg_int8', 'int8')]:
        r = runs.get(key)
        if not r:
            continue
        cur = r.get('curves', [])
        peak = max((c['val_combined'] or 0 for c in cur)) if cur else 0
        cells = ''
        for frac in [0.5, 0.8, 0.9, 0.95]:
            hit = next((c for c in cur if (c['val_combined'] or 0) >= frac * peak), None)
            cells += f'{(human_bytes(hit['uplink_bytes']) if hit else '-'):>14}'
        W(f'  {label:<16}{cells}')
    W("  (fraction of each scheme's own peak validation score)")
    W('\n  Accuracy vs. cumulative uplink (FedAvg)')
    W(f'  {'round':>7}{'uplink':>14}{'val detAcc':>12}{'val comb':>11}')
    cur = fa.get('curves', [])
    marks = [c for c in cur if c['round'] in {1, 5, 10, 20, 30, 40, 50, 60, 80, 100, len(cur)}]
    for c in marks:
        W(f'  {c['round']:>7}{human_bytes(c['uplink_bytes']):>14}{c['val_detect_acc'] or 0:>12.3f}{c['val_combined'] or 0:>11.4f}')
    results['expB'] = {'params_total': tot, 'params_encoder': enc, 'params_head': head, 'bytes_per_client_round_fedavg': per_round_client, 'bytes_per_client_round_fedper': enc * 4, 'uplink_total_fedavg': fedavg_total, 'uplink_total_fedper': fedper_total, 'vlm_params': VLM_PARAMS, 'reduction_vs_vlm_finetune': VLM_PARAMS / tot, 'tokens_bytes_per_image': TOKENS_PER_IMG}

def expC_byzantine(ModelCls, tr, va, te, device, cfg, results):
    W('\n' + '=' * 78)
    W('EXP-C  Byzantine robustness')
    W('=' * 78)
    clients, _, _ = build_partition(tr, va, te, cfg['client_split'], cfg['n_shards'], cfg['seed'])
    n_byz = cfg['n_byz']
    if cfg.get('byz_ids'):
        byz_ids = tuple((int(x) for x in cfg['byz_ids'].split(',') if x.strip() != ''))
    else:
        byz_ids = tuple(range(n_byz))
    ntot = sum((len(c['idx']) for c in clients))
    nbad = sum((len(clients[i]['idx']) for i in byz_ids))
    W(f'  {len(byz_ids)} of {len(clients)} clients are malicious: {[clients[i]['name'] for i in byz_ids]}')
    W(f'  they hold {nbad}/{ntot} = {nbad / ntot * 100:.1f}% of the training data, so size-weighted averaging is maximally exposed')
    results['expC_setup'] = {'byz_clients': [clients[i]['name'] for i in byz_ids], 'byz_data_share': nbad / ntot}
    R_byz = cfg.get('byz_rounds') or cfg['rounds']
    W(f'  byz_scale={cfg['byz_scale']}, rounds={R_byz}, trim={cfg['trim']}')
    attacks = list(cfg['attacks'])
    if 'none' not in attacks:
        attacks = ['none'] + attacks
    out = {}
    for attack in attacks:
        out[attack] = {}
        W(f'\n  --- attack: {attack} {('(no adversary: reference row)' if attack == 'none' else '')} ---')
        for rule in cfg['agg_rules']:
            tag = f'{attack}/{rule}'
            cfg_c = dict(cfg)
            cfg_c['fed_patience'] = max(cfg['fed_patience'], R_byz)
            r, _ = run_federated(ModelCls, tr, va, te, device, cfg_c, clients, algo='fedavg', agg_rule=rule, byz_ids=() if attack == 'none' else byz_ids, attack=attack, tag=tag, rounds=R_byz)
            out[attack][rule] = r
    results['expC'] = out
    W('\n  A trailing * marks a collapsed model: balanced accuracy at the 0.333 chance floor,\n  i.e. the model emits one class for every frame and has learned nothing.')
    W('  Numbers below are the FINAL-round global model. Under a successful attack no round\n  beats the random initialisation on validation, so best-val selection would just\n  return the untrained model; the final-round model is what a server really holds.')
    W('\n  Summary: detectability accuracy / balanced accuracy on XWOD test')
    header = f'  {'attack':<14}' + ''.join((f'{r:>15}' for r in cfg['agg_rules']))
    W(header)
    for attack in attacks:
        cells = ''
        for rule in cfg['agg_rules']:
            g = out[attack][rule].get('global_final') or out[attack][rule]['global']
            mark = '*' if g['detect_bal'] < 0.345 else ' '
            cells += f'{g['detect_acc']:>8.3f}/{g['detect_bal']:.3f}{mark}'
        W(f'  {attack:<14}{cells}')
    W('\n  Same table, severity accuracy')
    W(header)
    for attack in attacks:
        cells = ''
        for rule in cfg['agg_rules']:
            g = out[attack][rule].get('global_final') or out[attack][rule]['global']
            cells += f'{g['severity_acc']:>15.3f}'
        W(f'  {attack:<14}{cells}')
    W('\n  Same table, spatial (side) accuracy')
    W(header)
    for attack in attacks:
        cells = ''
        for rule in cfg['agg_rules']:
            g = out[attack][rule].get('global_final') or out[attack][rule]['global']
            cells += f'{g['side_acc']:>15.3f}'
        W(f'  {attack:<14}{cells}')
    W("\n  Attack results use the clean ('none') row of the same setting as the reference.")
    return out

def expP_participation(ModelCls, tr, va, te, device, cfg, results):
    """Evaluate partial participation and the corresponding communication cost."""
    W('\n' + '=' * 78)
    W('EXP-P  Dynamic participation (a fraction of the fleet reports each round)')
    W('=' * 78)
    clients, va_map, te_map = build_partition(tr, va, te, cfg['client_split'], cfg['n_shards'], cfg['seed'])
    W(f'  {len(clients)} agents; participation fraction C controls how many upload per round')
    out = {}
    for frac in cfg['fracs']:
        m = max(1, int(round(frac * len(clients))))
        cfg2 = dict(cfg)
        cfg2['frac'] = frac
        tag = f'FedAvg C={frac:.2f} ({m}/{len(clients)})'
        W(f'\n  -- {tag} --')
        r, _ = run_federated(ModelCls, tr, va, te, device, cfg2, clients, algo='fedavg', tag=tag)
        out[f'fedavg_C{frac:.2f}'] = r
    frac = min(cfg['fracs'])
    cfg2 = dict(cfg)
    cfg2['frac'] = frac
    W(f'\n  -- FedProx C={frac:.2f} (does the proximal term help when participation is sparse?) --')
    r, _ = run_federated(ModelCls, tr, va, te, device, cfg2, clients, algo='fedprox', tag=f'FedProx C={frac:.2f}')
    out[f'fedprox_C{frac:.2f}'] = r
    results['expP'] = out
    W('\n  Summary (XWOD global test set)')
    W(f'  {'method':<26}{'clients/rd':>12}{'uplink':>12}{'sevAcc':>9}{'detAcc':>9}{'sideAcc':>9}{'cntMAE':>9}')
    for key, r in out.items():
        g = r.get('global')
        if not g:
            continue
        fr = float(key.split('C')[-1])
        m = max(1, int(round(fr * len(clients))))
        W(f'  {key:<26}{f'{m}/{len(clients)}':>12}{human_bytes(r['uplink_bytes_total']):>12}{g['severity_acc']:>9.3f}{g['detect_acc']:>9.3f}{g['side_acc']:>9.3f}{g['count_mae']:>9.2f}')
    W('\n  Sparse participation costs accuracy but buys proportionally less traffic per round;\n  read this against the full-participation row of EXP-A.')
    return out

def expD_cross(ModelCls, device, cfg, results, args):
    W('\n' + '=' * 78)
    W('EXP-D  Cross-dataset federation (XWOD disaster fleet + BDD-IW routine fleet)')
    W('=' * 78)
    W('  Loading BDD-IW splits (memmap mode to limit RAM)...')
    b_tr = Split('BDD', 'train', args.bdd_tok_root, args.bdd_label_dir, 'memmap', device)
    b_va = Split('BDD', 'val', args.bdd_tok_root, args.bdd_label_dir, 'memmap', device)
    b_te = Split('BDD', 'test', args.bdd_tok_root, args.bdd_label_dir, 'memmap', device)
    W(f'  NOTE: the two fleets share the token geometry (192x4096) and the four task formats,\n        but differ in taxonomy and density ({'%.1f' % b_tr.density[:, 0].mean()} vs ~4.2 objects/img), which is the heterogeneity being tested.')
    results['expD'] = {'bdd_mean_objects': float(b_tr.density[:, 0].mean()), 'note': 'joint federation implemented as a separate driver; see report'}
    W('  EXP-D reports BDD fleet statistics only. Use the B2 configurations for BDD-IW federation.')
    return results['expD']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wf-dir', default=WF_DIR_DEFAULT)
    ap.add_argument('--tok-root', default='data/weather_patch_tokens')
    ap.add_argument('--label-dir', default='data/understanding_labels')
    ap.add_argument('--bdd-tok-root', default='data/bdd_patch_tokens')
    ap.add_argument('--bdd-label-dir', default='data/bdd_understanding_labels')
    ap.add_argument('--out-dir', default='runs/fed_results')
    ap.add_argument('--exp', default='0,A,B,C', help='0=sanity A=federation B=comms C=byzantine P=participation D=cross-dataset')
    ap.add_argument('--client-split', default='weather', choices=['weather', 'density', 'detect', 'random'], help="how the training set is partitioned into agents; use 'density' on BDD-IW, which has no weather labels")
    ap.add_argument('--n-shards', type=int, default=7, help='number of agents for --client-split density/random')
    ap.add_argument('--fracs', default='1.0,0.5,0.3', help='participation fractions swept by EXP-P')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--mode', default='auto', choices=['auto', 'gpu', 'ram', 'memmap'])
    ap.add_argument('--mem-budget-gb', type=float, default=0.0, help='override the RAM budget for cached tokens (GB); 0 = auto-detect from the cgroup limit')
    ap.add_argument('--gpu-reserve-gb', type=float, default=3.0, help='VRAM left free for activations')
    ap.add_argument('--no-progress', action='store_true')
    ap.add_argument('--rounds', type=int, default=60)
    ap.add_argument('--local-epochs', type=int, default=2)
    ap.add_argument('--cen-epochs', type=int, default=100)
    ap.add_argument('--frac', type=float, default=1.0, help='fraction of clients sampled per round')
    ap.add_argument('--mu', type=float, default=0.01, help='FedProx mu')
    ap.add_argument('--n-byz', type=int, default=2)
    ap.add_argument('--byz-scale', type=float, default=1.0)
    ap.add_argument('--trim', type=int, default=2, help='trimmed-mean: drop this many from each end')
    ap.add_argument('--byz-ids', default='', help='comma-separated client ranks to corrupt (0 = largest shard); default = the n_byz largest')
    ap.add_argument('--attacks', default='none,label_flip,sign_flip,gauss')
    ap.add_argument('--byz-rounds', type=int, default=0, help='rounds for EXP-C (0 = same as --rounds)')
    ap.add_argument('--agg-rules', default='fedavg,median,trimmed_mean,multi_krum')
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--eval-bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--lam', type=float, default=3.0)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--fed-patience', type=int, default=25)
    ap.add_argument('--log-every', type=int, default=10)
    ap.add_argument('--quick', action='store_true', help='tiny smoke run to verify wiring')
    args = ap.parse_args()
    global PROGRESS_ON
    PROGRESS_ON = not args.no_progress
    if args.quick:
        args.rounds, args.local_epochs, args.cen_epochs = (4, 1, 3)
        args.patience, args.fed_patience, args.log_every = (99, 99, 1)
        args.attacks = 'none,sign_flip'
        args.agg_rules = 'fedavg,multi_krum'
        args.seeds = [0]
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    W('=' * 78)
    W('Federated PaDU — experiments on cached frozen-VLM tokens')
    W('=' * 78)
    W(f'time        : {time.strftime('%Y-%m-%d %H:%M:%S')}')
    W(f'device      : {device}' + (f'  ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1000000000.0:.0f} GB)' if device.type == 'cuda' else ''))
    if device.type != 'cuda':
        W('  !! torch.cuda.is_available() is False — this will be very slow.')
        W(f'  !! torch {torch.__version__}, built for CUDA {torch.version.cuda}. Check `nvidia-smi` and that the container has a GPU attached.')
    lim, use, src = mem_limits()
    W(f'memory      : limit {human_bytes(lim)}, in use {human_bytes(use)} [{src}]')
    W(f'              budget for tokens: {human_bytes(usable_ram_bytes(args.mem_budget_gb))}')
    W(f'exp         : {args.exp}   seeds: {args.seeds}' + ('   [QUICK SMOKE RUN]' if args.quick else ''))
    W(f'rounds      : {args.rounds} x {args.local_epochs} local epochs, frac={args.frac}')
    W(f'partition   : {args.client_split}' + (f' ({args.n_shards} shards)' if args.client_split in ('density', 'random') else ''))
    W(f'labels      : {args.label_dir}')
    ModelCls = import_model(args.wf_dir)

    def split_bytes(root, sp):
        p = os.path.join(root, f'{sp}_tokens.npy')
        return os.path.getsize(p) if os.path.exists(p) else 0
    sizes = {s: split_bytes(args.tok_root, s) for s in ['train', 'val', 'test']}
    plan, note = plan_placement(args.mode, sizes, device, args.mem_budget_gb, args.gpu_reserve_gb)
    W(f'token store : total {human_bytes(sum(sizes.values()))}   ({note})')
    for k in ['train', 'val', 'test']:
        W(f'              {k:<6} {human_bytes(sizes[k]):>10}  -> {plan[k]}')
    if all((v == 'memmap' for v in plan.values())):
        W(f'  !! everything is on memmap: each epoch re-reads {human_bytes(sizes['train'])} from disk. Consider --mem-budget-gb or freeing memory.')
    mode = plan['train']
    W('\nLoading XWOD splits ...')
    tr = Split('XWOD', 'train', args.tok_root, args.label_dir, plan['train'], device)
    va = Split('XWOD', 'val', args.tok_root, args.label_dir, plan['val'], device)
    te = Split('XWOD', 'test', args.tok_root, args.label_dir, plan['test'], device)
    exps = [e.strip().upper() for e in args.exp.split(',')]
    all_results = {'args': vars(args), 'mode': mode, 'seeds': {}}
    for seed in args.seeds:
        W('\n' + '#' * 78)
        W(f'#  SEED {seed}')
        W('#' * 78)
        cfg = dict(seed=seed, d_model=384, n_layers=3, n_heads=6, dropout=0.1, lr=args.lr, wd=args.wd, lam=args.lam, bs=args.bs, eval_bs=args.eval_bs, patience=args.patience, cen_epochs=args.cen_epochs, rounds=args.rounds, local_epochs=args.local_epochs, frac=args.frac, mu=args.mu, fed_patience=args.fed_patience, log_every=args.log_every, trim=args.trim, byz_scale=args.byz_scale, client_split=args.client_split, n_shards=args.n_shards, fracs=[float(x) for x in args.fracs.split(',')], n_byz=args.n_byz, byz_ids=args.byz_ids, byz_rounds=args.byz_rounds, attacks=[a.strip() for a in args.attacks.split(',')], agg_rules=[r.strip() for r in args.agg_rules.split(',')])
        res = {}
        cen_test = None
        if '0' in exps:
            cen_test = strip_metrics(exp0_sanity(ModelCls, tr, va, te, device, cfg, res))
        if 'A' in exps:
            if cen_test is None:
                _, mt = run_centralized(ModelCls, tr, va, te, device, cfg)
                cen_test = strip_metrics(mt)
                res['exp0_centralized'] = cen_test
            runs = expA_federation(ModelCls, tr, va, te, device, cfg, res, cen_test)
            if 'B' in exps:
                expB_communication(ModelCls, device, cfg, res, runs)
        if 'C' in exps:
            expC_byzantine(ModelCls, tr, va, te, device, cfg, res)
        if 'P' in exps:
            expP_participation(ModelCls, tr, va, te, device, cfg, res)
        if 'D' in exps:
            expD_cross(ModelCls, device, cfg, res, args)
        all_results['seeds'][str(seed)] = res
        with open(os.path.join(args.out_dir, 'fed_results.json'), 'w') as f:
            json.dump(all_results, f, indent=1, default=str)
        flush_log(os.path.join(args.out_dir, 'FED_REPORT.txt'))
    csv_path = os.path.join(args.out_dir, 'fed_curves.csv')
    with open(csv_path, 'w') as f:
        f.write('seed,method,round,uplink_bytes,val_combined,val_detect_acc,val_severity_acc,val_side_acc\n')
        for sd, res in all_results['seeds'].items():
            for grp in ['expA', 'expC']:
                blob = res.get(grp, {})
                items = []
                if grp == 'expA':
                    items = [(k, v) for k, v in blob.items()]
                else:
                    for atk, d in blob.items():
                        for rule, v in d.items():
                            items.append((f'{atk}:{rule}', v))
                for name, r in items:
                    for c in r.get('curves', []):
                        f.write(f'{sd},{name},{c['round']},{c['uplink_bytes']},{c['val_combined']},{c['val_detect_acc']},{c['val_severity_acc']},{c['val_side_acc']}\n')
    W('\n' + '=' * 78)
    W('DONE')
    W('=' * 78)
    W(f'  report : {os.path.join(args.out_dir, 'FED_REPORT.txt')}')
    W(f'  json   : {os.path.join(args.out_dir, 'fed_results.json')}')
    W(f'  curves : {csv_path}')
    flush_log(os.path.join(args.out_dir, 'FED_REPORT.txt'))
from pathlib import Path
import hashlib

def _revision_save(scope, kind):
    from pathlib import Path
    import re

    def cpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        if isinstance(x, dict):
            return {k: cpu(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [cpu(v) for v in x]
        return x
    cfg = scope['cfg']
    tag = re.sub('[^a-zA-Z0-9_-]', '_', scope.get('tag', kind))
    target = Path(os.environ['PADU_RUN_OUT']) / 'checkpoints'
    target.mkdir(exist_ok=True)
    path = target / f'seed{cfg['seed']}_{tag}.pt'
    if path.exists():
        raise RuntimeError('Refusing checkpoint overwrite: ' + str(path))
    bundle = {'config': cfg, 'tag': scope.get('tag'), 'kind': kind, 'best_state': scope.get('best_state'), 'best_private': scope.get('best_private'), 'final_private': scope.get('private'), 'best_round': scope.get('best_round'), 'best_val': scope.get('best'), 'curves': scope.get('curves'), 'final_state': scope.get('final_state')}
    model = scope.get('model', scope.get('gmodel'))
    if model is not None:
        bundle['returned_model_state'] = model.state_dict()
    torch.save(cpu(bundle), path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.sha256').write_text(h + '  ' + path.name + '\n')
    print('REVISION_CHECKPOINT', path, h, flush=True)
_revision_original_partition = build_partition

def build_partition(*args, **kwargs):
    import inspect
    result = _revision_original_partition(*args, **kwargs)

    def simple(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, dict):
            return {str(k): simple(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [simple(v) for v in x]
        return x
    params = inspect.signature(_revision_original_partition).bind(*args, **kwargs)
    settings = {k: simple(v) for k, v in params.arguments.items() if k not in ('tr', 'va', 'te')}
    blob = {'settings': settings, 'partition_positions': simple(result), 'note': 'Indices refer to the archived, unchanged cache metadata; original split limitations remain.'}
    raw = json.dumps(blob, sort_keys=True, default=str).encode()
    folder = Path(os.environ['PADU_RUN_OUT']) / 'partitions'
    folder.mkdir(exist_ok=True)
    (folder / (hashlib.sha256(raw).hexdigest() + '.json')).write_bytes(raw)
    return result
if __name__ == '__main__':
    main()