#!/usr/bin/env python3
"""
Minimal playground to understand how D-Intervention / ReFT works.

Run from D-Intervention dir (so local pyreft and task_config are found):
  conda activate bo-intervention
  python playground.py

Flow: load base model -> define where to intervene -> wrap in ReftModel -> forward pass.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# 1. Where interventions live: pyreft/interventions.py
#    Each class (LoreftIntervention, RedIntervention, VIBRawreftIntervention, ...)
#    implements forward(base, source=None, subspaces=None) and operates on
#    hidden states at a chosen layer.
# -----------------------------------------------------------------------------
from pyreft import (
    get_reft_model,
    ReftConfig,
    LoreftIntervention,   # LoReFT: h + R^T(Wh + b - Rh)
    RedIntervention,      # RED: rotation + bias
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Small model from /datasets/ai/ (TinyLlama not present; using Llama-3.2-1B from llama3 cache)
DATASETS_AI = "/datasets/ai/llama3/hub"
# Llama-3.2-1B base model (1B params, similar size to TinyLlama)
MODEL_NAME = f"{DATASETS_AI}/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"

RANK = 4
LAYER = 0


def main():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE == "cpu":
        model = model.to(DEVICE)

    config = model.config
    hidden_size = config.hidden_size
    n_layers = config.n_layer if hasattr(config, "n_layer") else config.num_hidden_layers

    # -------------------------------------------------------------------------
    # 2. ReftConfig: "where" to intervene (which layer/component) and "what"
    #    (which intervention module). Same idea as in train.py ~352-380.
    #    component "block_output" = output of that transformer block (pyvene convention).
    # -------------------------------------------------------------------------
    representations = [
        {
            "layer": LAYER,
            "component": "block_output",
            "low_rank_dimension": RANK,
            "intervention": LoreftIntervention(
                embed_dim=hidden_size,
                low_rank_dimension=RANK,
                dropout=0.0,
                dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                device=DEVICE,
            ),
        }
    ]

    reft_config = ReftConfig(representations=representations)
    reft_model = get_reft_model(model, reft_config, set_device=True)

    # -------------------------------------------------------------------------
    # 3. Inspect: only the intervention params are trainable; base model is frozen.
    # -------------------------------------------------------------------------
    reft_model.print_trainable_parameters()

    # -------------------------------------------------------------------------
    # 4. Single forward pass to see that the wrapped model runs.
    #    reft_model.forward() uses pyvene under the hood to run the base model
    #    and apply interventions at the specified positions.
    # -------------------------------------------------------------------------
    prompt = "Generate a word"
    inputs = tokenizer(prompt, return_tensors="pt").to(reft_model.model.device)
    # pyvene IntervenableModel expects (inputs_dict, unit_locations=...); returns (base_outputs, cf_outputs)
    seq_len = inputs["input_ids"].shape[-1]
    intervention_locations = [[[i for i in range(seq_len)]]]  # [layers][batch][positions]
    with torch.no_grad():
        _, out = reft_model(
            {"input_ids": inputs["input_ids"], "attention_mask": inputs.get("attention_mask")},
            unit_locations={"sources->base": (None, intervention_locations)},
        )

    logits = out.logits
    print(f"Forward pass OK. Logits shape: {logits.shape}")

    # Optional: one training step (so you can set breakpoints and step through)
    reft_model.model.train()
    reft_model.model.eval()  # keep eval for this demo
    # To actually train, you'd use ReftTrainerForCausalLM as in train.py, or
    # call reft_model() with labels and compute loss yourself.

    # print("\nDone. Next steps to explore:")
    # print("  - pyreft/interventions.py  : each intervention's forward()")
    # print("  - pyreft/reft_model.py      : ReftModel wraps pyvene.IntervenableModel")
    # print("  - pyreft/reft_trainer.py    : ReftTrainerForCausalLM training loop")
    # print("  - train.py (finetune())     : full pipeline: data -> ReftConfig -> train")


if __name__ == "__main__":
    main()
