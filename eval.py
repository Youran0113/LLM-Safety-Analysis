import torch, json
import matplotlib.pyplot as plt
from transformers import AutoConfig
from tqdm import tqdm
from utils import tokenizer, build_input, model, acts

L = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").num_hidden_layers
ds = json.load(open("data.json"))
vecs = torch.load("vecs_0.pt", weights_only=False)

comps = ["attn", "mlp", "residual"]
methods = list(vecs.keys())
b_list = list(vecs[methods[0]].keys())

feats = []
for task, pairs in tqdm(ds["test"].items(), desc="Extract"):
    for p in pairs:
        acts.clear()
        blk = ds["data"][p["pos"]]
        txt = build_input(blk)
        pref = f"<｜User｜>{blk['prompt']}<｜Assistant｜><think>{blk['context']} " if blk["context"] else f"<｜User｜>{blk['prompt']}<｜Assistant｜><think>"
        
        ids = tokenizer(txt, return_tensors="pt")["input_ids"].to("cuda")
        st, ed = len(tokenizer(pref)["input_ids"]), ids.shape[1]
        
        with torch.no_grad(): model(ids)
        
        feats.append((task, {l: {c: acts[f"L{l}_{c}"][0, st:ed, :].mean(0).clone().float() for c in comps} for l in range(L)}))

res = {}
for m in tqdm(methods, desc="Eval"):
    for l in tqdm(range(L), leave=False):
        for c in comps:
            if c not in vecs[m].get(b_list[0], {}).get(l, {}):
                res[(m, l, c)] = {"t1": 0, "t3": 0, "jac": 0}
                continue
                
            t1, t3, jac, tot = 0, 0, 0.0, len(feats)
            for true_b, h_dict in feats:
                h = h_dict[l][c].cuda()
                scores = {b: (torch.dot(h, vecs[m][b][l][c].cuda()) / (vecs[m][b][l][c].norm() + 1e-8)).item() for b in b_list}
                
                ranked = sorted(scores, key=scores.get, reverse=True)
                if ranked[0] == true_b: t1 += 1
                if true_b in ranked[:3]: 
                    t3 += 1
                    jac += 1/3
            
            res[(m, l, c)] = {"t1": t1/tot, "t3": t3/tot, "jac": jac/tot}

fig, axs = plt.subplots(3, len(methods), figsize=(5*len(methods), 10), sharey="row")
if len(methods) == 1: axs = axs.reshape(3, 1)

m_keys, m_lbls = ["t1", "t3", "jac"], ["Top-1", "Top-3", "Jaccard"]

for i, m in enumerate(methods):
    for j, mk in enumerate(m_keys):
        ax = axs[j, i]
        for c in comps:
            ax.plot(range(L), [res[(m, l, c)][mk] for l in range(L)], label=c)
        ax.set_xlabel("Layer")
        if j == 0: ax.legend()
        if i == 0: ax.set_ylabel(m_lbls[j])
        
        best = max((res[(m, l, c)][mk], l, c) for l in range(L) for c in comps)
        ax.set_title(f"{m.upper()} {m_lbls[j]} (L{best[1]}-{best[2]}={best[0]:.2f})")

bt1 = max(res, key=lambda k: res[k]["t1"])
bt3 = max(res, key=lambda k: res[k]["t3"])

plt.suptitle(f"Best T1: {bt1[0]} L{bt1[1]}-{bt1[2]} ({res[bt1]['t1']:.3f}) | Best T3: {bt3[0]} L{bt3[1]}-{bt3[2]} ({res[bt3]['t3']:.3f})")
plt.tight_layout()
plt.savefig("eval.png", dpi=150)
plt.close()

print(f"Top-1 Winner: {bt1[0].upper()} L{bt1[1]} ({bt1[2]}) -> {res[bt1]['t1']:.4f}")
print(f"Top-3 Winner: {bt3[0].upper()} L{bt3[1]} ({bt3[2]}) -> {res[bt3]['t3']:.4f}")