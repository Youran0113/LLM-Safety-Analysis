import os, sys, json, torch
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def calc_rs(P, N):
    return P.mean(0) - N.mean(0)

def calc_svd(P, N, r=64):
    v = calc_rs(P, N)
    Pc, Nc = P - P.mean(0, keepdim=True), N - N.mean(0, keepdim=True)
    uP, _, _ = torch.linalg.svd(Pc.T, full_matrices=False)
    uN, _, _ = torch.linalg.svd(Nc.T, full_matrices=False)
    piP, piN = uP[:, :r] @ uP[:, :r].T, uN[:, :r] @ uN[:, :r].T
    return (torch.eye(v.shape[0], device=v.device) - piN) @ piP @ v

def calc_knn(P, N, k=5):
    tot = P.shape[0] + N.shape[0]
    cat = torch.cat([P, N], dim=0)
    dist = torch.cdist(cat, cat, p=2)
    dist.fill_diagonal_(float('inf'))
    
    _, idx = dist.topk(min(k, tot - 1), dim=1, largest=False)
    y = torch.cat([torch.ones(len(P), device=P.device), torch.zeros(len(N), device=N.device)])
    
    match = (y[idx] == y.unsqueeze(1)).float()
    purity = match.mean(1)
    
    wP, wN = purity[:len(P)], purity[len(P):]
    cP = (wP.unsqueeze(1) * P).sum(0) / (wP.sum() + 1e-8)
    cN = (wN.unsqueeze(1) * N).sum(0) / (wN.sum() + 1e-8)
    return cP - cN

def calc_rfm(P, N, ep=5, L=1.0, lam=1e-3):
    cat = torch.cat([P, N], dim=0).double()
    n, d = cat.shape
    y = torch.cat([torch.ones(len(P)), torch.zeros(len(N))]).to(cat.device).double()
    M = torch.eye(d, device=cat.device, dtype=torch.float64)

    for _ in range(ep):
        Y = cat @ M
        diag = (cat * Y).sum(1)
        dist = (diag[:, None] + diag[None, :] - 2 * (Y @ cat.T)).clamp(min=0).sqrt()
        K = torch.exp(-dist / L)
        
        A = K + lam * torch.eye(n, device=cat.device, dtype=torch.float64)
        coef = torch.linalg.solve(A, y)
        
        w = (coef[None, :] * K) / dist.clamp(min=1e-10)
        w.fill_diagonal_(0.0)
        
        grad = -(1.0 / L) * (Y * w.sum(1)[:, None] - w @ Y)
        M = (grad.T @ grad) / n

    M += 1e-6 * torch.eye(d, device=M.device, dtype=M.dtype)
    v = M @ (P.mean(0).double() - N.mean(0).double())
    return (v / (v.norm() + 1e-8)).float()

gid = int(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)
gpus = sorted(int(x) for x in sys.argv[2:])
rank, world = gpus.index(gid), len(gpus)

torch.cuda.set_device(0)
dev = "cuda:0"
mdl_name, data_key = "Qwen/Qwen3-8B", "qwen3-8b"
L = AutoConfig.from_pretrained(mdl_name).num_hidden_layers

print(f"[GPU {gid}] Init {mdl_name}...")
model = AutoModelForCausalLM.from_pretrained(mdl_name).to(dev)
tok = AutoTokenizer.from_pretrained(mdl_name)
cache = {}

def hook(name):
    def _h(m, i, o):
        out = o[0] if isinstance(o, (tuple, list)) else o
        if out.requires_grad: out.retain_grad()
        cache[name] = out
    return _h

for i, layer in enumerate(model.model.layers):
    layer.self_attn.o_proj.register_forward_hook(hook(f"L{i}_attn"))
    layer.mlp.register_forward_hook(hook(f"L{i}_mlp"))
    layer.register_forward_hook(hook(f"L{i}_residual"))

def get_acts(blk):
    tgt = blk["target_sentence"]
    ctx = f"{blk['context']} " if blk["context"] else ""
    txt = f"<|im_start|>user\n{blk['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n{ctx}{tgt}\n</think>\n\n<|im_end|>\n"
    
    ed_char = txt.find(tgt) + len(tgt)
    enc = tok(txt, return_tensors="pt", return_offsets_mapping=True)
    ids, off = enc["input_ids"][0], enc["offset_mapping"][0]
    
    match = [i for i, (s, e) in enumerate(off.tolist()) if s < ed_char and e > txt.find(tgt)]
    st, ed = (match[0], match[-1] + 1) if match else (len(ids)-1, len(ids))
    
    cache.clear()
    model.zero_grad()
    loss = -torch.log_softmax(model(ids.unsqueeze(0).to(dev)).logits[0, -1, :], dim=-1).max()
    loss.backward()
    
    return {l: {c: cache[f"L{l}_{c}"][0, st:ed, :].mean(0).detach().cpu().clone() for c in ["attn", "mlp", "residual"]} for l in range(L)}

ds = json.load(open("data.json"))
val_ids = {i for i, x in enumerate(ds["data"]) if x["model"] == data_key}
tasks = {k: [p for p in v if p["pos"] in val_ids and p["neg"] in val_ids] for k, v in ds["train"].items()}

my_tasks = [t for i, t in enumerate(sorted(tasks.keys())) if i % world == rank]
vecs = lambda: defaultdict(lambda: defaultdict(dict))
out = {"rs": vecs(), "svd": vecs(), "knn": vecs(), "rfm": vecs()}

for t in tqdm(my_tasks, desc=f"GPU {gid}"):
    pos = [get_acts(ds["data"][p["pos"]]) for p in tasks[t]]
    neg = [get_acts(ds["data"][p["neg"]]) for p in tasks[t]]
    
    for l in range(L):
        for c in ["attn", "mlp", "residual"]:
            P = torch.stack([x[l][c] for x in pos]).float().cuda()
            N = torch.stack([x[l][c] for x in neg]).float().cuda()
            
            out["rs"][t][l][c] = calc_rs(P, N).cpu()
            out["svd"][t][l][c] = calc_svd(P, N).cpu()
            out["knn"][t][l][c] = calc_knn(P, N).cpu()
            try:
                out["rfm"][t][l][c] = calc_rfm(P, N).cpu()
            except:
                out["rfm"][t][l][c] = out["rs"][t][l][c]

    torch.cuda.empty_cache()

clean_out = {m: {t: {l: dict(c) for l, c in lv.items()} for t, lv in tv.items()} for m, tv in out.items()}
torch.save(clean_out, f"vecs_{gid}.pt")