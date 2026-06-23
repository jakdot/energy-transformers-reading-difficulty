"""
Check that your downloaded NRGPT model is sensible, i.e., it generates an ok text.
"""

import sys
import torch
import tiktoken

sys.path.insert(0, "nrgpt")
sys.path.insert(0, "nrgpt/models")

from model_config import ModelConfig
from models.energy_models import NRGPT_H_FF2W

CKPT = "nrgpt/out-OWT02_owt_best_configs/Best_OWT02_owt_best_configs_model=NRGPT_H_FF2W_embed=1536_depth=6_heads=12_LR=3e-05_minLR=None_minLrDiv=10.0_numIter=100000_exp_kko52p3j.pt" # path to NRGPT model

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(CKPT, map_location=device)
config = ModelConfig(**checkpoint["model_args"])
model = NRGPT_H_FF2W(config).to(device)
state_dict = checkpoint["model"]
for prefix in ["_orig_mod.", "module."]:
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded. Generating...")

enc = tiktoken.get_encoding("gpt2")
prompt = "In recent years, researchers have discovered that "
idx = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    out = model.generate(idx, max_new_tokens=50, greedy=False, temperature=0.8)

print(enc.decode([t for t in out[0].cpu().tolist() if t <= enc.max_token_value]))
