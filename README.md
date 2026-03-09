# Disentangling Safety and Utility in LLM Activation Spaces

**CS 639 Project — Group 14**
Sungjin Cheong · Enze Zhang · Youran Wang · Yuliang Wu
*Equal contribution*

---

## Overview

We analyze the geometric structure of safety-behavior steering vectors across multiple LLM architectures. Using contrastive activation pairs from the [reasoning-safety-behaviours](https://huggingface.co/datasets/AISafety-Student/reasoning-safety-behaviours) dataset (20 behaviors), we extract steering vectors and measure cross-model representational similarity to ask: **how universal is safety?**

---

## Models

| Model | Architecture | Size |
|---|---|---|
| DeepSeek-R1-Distill-Llama-8B | Llama | 8B |
| DeepSeek-R1-0528-Qwen3-8B | Qwen3 | 8B |
| Qwen3-8B | Qwen3 | 8B |
| DeepSeek-R1-Distill-Qwen-32B | Qwen | 32B |

---

## Methods

**Steering vector extraction:** DoM, SVD, KNN, RFM

**Cross-model geometry:** CKA, Procrustes, RSA

---

## Usage

```bash
# Extract steering vectors (run one process per GPU)
python work_v2.py <gpu_id> <gpu0> <gpu1> ...

# Example: 3 GPUs
python work_v2.py 0 0 1 2 &
python work_v2.py 1 0 1 2 &
python work_v2.py 2 0 1 2 &
```

Output: `steering_<gpu_id>.safetensors` — keys formatted as `method|behavior|layer|component`

---

## Requirements

```
torch
transformers
safetensors
tqdm
```

---

## Paper

See `Team14_Proposal.pdf` for the full project proposal (ACL format).
