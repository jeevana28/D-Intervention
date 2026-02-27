#!/usr/bin/env python3
"""
Semantle word search with LoReFT/Distributional interventions (training only).

Intervention types
------------------
LoreftWordIntervention         : point-estimate b per word (float vector subspaces).
DistributionalWordIntervention : distribution N(mu_w, sigma_w^2) per word (word IDs as
                                 subspaces, KL regularisation via --beta, semantic
                                 structure loss via --lambda_sem).

Usage
-----
python semantle_reft_bo.py --mode train \
    --semantle_csv /path/to/mob.csv --top_k 50 \
    --model meta-llama/Llama-3.2-1B --cache_dir /datasets/ai/llama3/hub \
    --intervention_type DistributionalWordIntervention \
    --beta 0.5 --lambda_sem 0.0 \
    --layer 15 --low_rank_dim 8 --epochs 30 \
    --output_dir ./out_distributional

# Or use the provided shell script:
#   bash run_distributional.sh
"""

import os
import csv
import json
import argparse
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from task_config import task_config
from dataset import LoReftSemantleDataset
from pyreft import (
    get_reft_model,
    ReftConfig,
    LoreftWordIntervention,
    DistributionalWordIntervention,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainingArguments,
)
from pyreft.reft_trainer import ReftTrainerForCausalLM
from datasets import Dataset as HFDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEMANTLE_PROMPT = task_config["semantle"]["fixed_prompt"]

INTERVENTION_TYPES = {
    "LoreftWordIntervention": LoreftWordIntervention,
    "DistributionalWordIntervention": DistributionalWordIntervention,
}


# ---------------------------------------------------------------------------
# Custom data collator
# ---------------------------------------------------------------------------

@dataclass
class SemantleDataCollator:
    """
    Wraps DataCollatorForSeq2Seq but manually handles subspaces,
    intervention_locations, and id so HF padding never sees them.

    Works for both float subspaces (LoreftWordIntervention) and long subspaces
    (DistributionalWordIntervention word IDs): torch.stack preserves dtype.
    """
    tokenizer: object
    model: object

    def __post_init__(self):
        self.inner = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            padding="longest",
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = [deepcopy(inst) for inst in instances]
        subspaces_list = [inst.pop("subspaces") for inst in instances
                          if "subspaces" in inst]
        intloc_list    = [inst.pop("intervention_locations") for inst in instances
                          if "intervention_locations" in inst]
        _ = [inst.pop("id", None) for inst in instances]

        batch = self.inner(instances)
        max_seq_len = batch["input_ids"].shape[-1]

        if subspaces_list:
            stacked = []
            for s in subspaces_list:
                if not isinstance(s, torch.Tensor):
                    s = torch.tensor(s)
                stacked.append(s)
            batch["subspaces"] = torch.stack(stacked, dim=0)

        if intloc_list:
            loc_tensor = torch.tensor(intloc_list, dtype=torch.long)
            batch["intervention_locations"] = loc_tensor[..., :max_seq_len]

        return batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_semantle_csv(
    csv_path: str, top_k: int = 100
) -> Tuple[str, List[str], Dict[str, float]]:
    """Load a Semantle CSV (Word, Similarity). Returns (target_word, words, sim_map)."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["Word"].strip(), float(row["Similarity"])))
    rows.sort(key=lambda x: -x[1])
    target_word = rows[0][0]
    top = rows[:top_k]
    return target_word, [r[0] for r in top], {r[0]: r[1] for r in top}


# def build_word_sim_matrix(
#     words: List[str], sim_map: Dict[str, float]
# ) -> torch.Tensor:
#     """
#     N x N target similarity matrix for the semantic structure loss.
#     sim_matrix[i][j] = sim[i] * sim[j]  (both measured vs. the target word).
#     """
#     sims = torch.tensor([sim_map.get(w, 0.0) for w in words], dtype=torch.float32)
#     mat = sims.unsqueeze(1) * sims.unsqueeze(0)
#     mat.fill_diagonal_(1.0)
#     return mat


def build_semantle_data(
    words: List[str],
    b_vectors: Optional[Dict[str, np.ndarray]] = None,
    low_rank_dim: int = 8,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Build {word, subspaces} dicts for LoReftSemantleDataset.
    use_word_ids=True  -> subspaces = [word_id]  (DistributionalWordIntervention)
    """
    word_to_id = {w: i for i, w in enumerate(words)}
    data = []
    for w in words:
        item = {"word": w}
        item["subspaces"] = [word_to_id[w]]
        data.append(item)
    return data, word_to_id


# ---------------------------------------------------------------------------
# Qualitative + quantitative evaluation callback
# ---------------------------------------------------------------------------

def _get_embed_model():
    """Lazy-load sentence-transformer for embedding similarity (optional)."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        return None


class QualitativeEvalCallback(TrainerCallback):
    """Every eval_steps, print model predictions and embedding similarity (target vs generated)."""

    def __init__(self, intervenable, tokenizer, words, prompt, eval_steps=100):
        self.intervenable = intervenable
        self.tokenizer    = tokenizer
        self.words        = words
        self.prompt       = prompt
        self.eval_steps   = eval_steps
        self._embed_model = None
        n = len(words)
        self.sample_indices = list(range(0, n, max(1, n // 8)))[:8]

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return

        iv = list(self.intervenable.interventions.values())[0]
        if isinstance(iv, (list, tuple)):
            iv = iv[0]

        device    = "cuda" if torch.cuda.is_available() else "cpu"
        enc       = self.tokenizer(self.prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        last_pos  = input_ids.shape[1] - 1
        unit_locs = [[[last_pos]]]   # (num_interventions=1, batch=1, n_pos=1)

        targets, generated_list = [], []
        print(f"\n[QualEval] step {state.global_step}")
        print(f"  {'target':<20}  generated")
        self.intervenable.eval()
        with torch.no_grad():
            for idx in self.sample_indices:
                subspaces = [[[idx]]]   # word ID works for both intervention types
                base_out, cf_out = self.intervenable(
                    {"input_ids": input_ids, "attention_mask": attn_mask},
                    unit_locations={"sources->base": (None, unit_locs)},
                    subspaces=subspaces,
                )
                logits    = (cf_out if cf_out is not None else base_out).logits[0, -1].float()
                generated = self.tokenizer.decode([logits.argmax().item()]).strip()
                targets.append(self.words[idx])
                generated_list.append(generated)
                print(f"  {self.words[idx]:<20}  {generated!r}")
        self.intervenable.train()

        # Quantitative: embedding similarity (target vs generated)
        if self._embed_model is None:
            self._embed_model = _get_embed_model()
        if self._embed_model is not None:
            emb_tgt = self._embed_model.encode(targets, normalize_embeddings=True)
            emb_gen = self._embed_model.encode(generated_list, normalize_embeddings=True)
            sims   = np.sum(emb_tgt * emb_gen, axis=1)   # cosine sim (embeddings normalized)
            mean_sim = float(np.mean(sims))
            print(f"  embed_sim (target vs generated): {mean_sim:.4f} (mean over sample)")
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"eval/embed_sim": mean_sim, "train/global_step": state.global_step})
            except ImportError:
                pass
        else:
            print("  (install sentence-transformers for embed_sim metric)")
        print()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_semantle_reft(
    model_name: str,
    words: List[str],
    intervention_type: str = "LoreftWordIntervention",
    b_vectors: Optional[Dict[str, np.ndarray]] = None,
    word_sim_matrix: Optional[torch.Tensor] = None,
    low_rank_dim: int = 8,
    layer: int = 15,
    position: str = "l1",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.06,   # converted to warmup_steps internally
    beta: float = 0.5,
    lambda_sem: float = 0.0,
    output_dir: str = "./semantle_reft_out",
    cache_dir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    seed: int = 42,
    eval_steps: int = 100,
) -> str:
    """Train LoreftWordIntervention or DistributionalWordIntervention. Returns output_dir."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="right", use_fast=False, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        cache_dir=cache_dir,
    )

    hidden_size = model.config.hidden_size
    iv_cls = INTERVENTION_TYPES[intervention_type]
    iv_kwargs = dict(
        embed_dim=hidden_size,
        low_rank_dimension=low_rank_dim,
        dropout=0.0,
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device=DEVICE,
        num_words=len(words),
    )
    if intervention_type == "DistributionalWordIntervention":
        iv_kwargs["beta"] = beta

    reft_config = ReftConfig(representations=[{
        "layer": layer,
        "component": "block_output",
        "low_rank_dimension": low_rank_dim,
        "intervention": iv_cls(**iv_kwargs),
    }])
    reft_model = get_reft_model(model, reft_config, set_device=True)
    reft_model.print_trainable_parameters()

    data, _ = build_semantle_data(
        words,
        b_vectors=None,
        low_rank_dim=low_rank_dim,
    )
    
    train_dataset = LoReftSemantleDataset(
        task="semantle", data_path="semantle", tokenizer=tokenizer,
        data_split="train", dataset=HFDataset.from_list(data),
        seed=seed, max_n_example=len(data),
        num_interventions=1, position=position,
        share_weights=False, low_rank_dimension=low_rank_dim,
    )

    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        _report_to = "wandb"
    else:
        _report_to = "none"

    total_steps = max(1, (len(words) // batch_size) * epochs)
    warmup_steps = int(warmup_ratio * total_steps)
    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, learning_rate=lr,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        logging_steps=10, save_strategy="no",
        remove_unused_columns=False,
        report_to=_report_to,
        run_name=wandb_run_name,
    )
    qual_eval_cb = QualitativeEvalCallback(
        intervenable=reft_model,
        tokenizer=tokenizer,
        words=words,
        prompt=SEMANTLE_PROMPT,
        eval_steps=eval_steps,
    )

    trainer = ReftTrainerForCausalLM(
        model=reft_model, tokenizer=tokenizer, args=training_args,
        train_dataset=train_dataset,
        data_collator=SemantleDataCollator(tokenizer=tokenizer, model=model),
        word_sim_matrix=word_sim_matrix,
        lambda_sem=lambda_sem,
        callbacks=[qual_eval_cb],
    )
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    reft_model.save_intervention(
        save_directory=os.path.join(output_dir, "intervenable_model"),
        include_model=False,
    )
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "words.json"), "w") as f:
        json.dump(words, f)
    print(f"[train] Saved to {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Semantle LoReFT training (LoreftWord / Distributional)")
    parser.add_argument("--mode", choices=["train"], default="train")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace model cache directory.")

    wg = parser.add_mutually_exclusive_group()
    wg.add_argument("--words", type=str, nargs="+",
                    default=["computer", "laptop", "keyboard", "screen", "mouse"])
    wg.add_argument("--semantle_csv", type=str, default=None,
                    help="Semantle CSV (Word,Similarity). Top row is the target.")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Use the top-k words from the CSV.")

    parser.add_argument("--intervention_type",
                        choices=list(INTERVENTION_TYPES.keys()),
                        default="LoreftWordIntervention")
    parser.add_argument("--low_rank_dim", type=int, default=8)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--position", type=str, default="l1",
                        help="Intervention position, e.g. 'l1', 'f1', 'f1+l1'.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        choices=["linear", "cosine", "cosine_with_restarts",
                                 "polynomial", "constant", "constant_with_warmup"],
                        help="LR scheduler. Default: linear decay to 0.")
    parser.add_argument("--warmup_ratio", type=float, default=0.06,
                        help="Fraction of steps used for LR warmup. Default: 0.06.")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="KL weight for DistributionalWordIntervention.")
    parser.add_argument("--lambda_sem", type=float, default=0.0,
                        help="Semantic structure loss weight.")
    parser.add_argument("--output_dir", type=str, default="./semantle_reft_out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name. Omit to disable W&B logging.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (optional).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #ignore sim_map for now
    sim_map: Dict[str, float] = {}
    if args.semantle_csv:
        _, words, sim_map = load_semantle_csv(args.semantle_csv, top_k=args.top_k)
        print(f"[data] {len(words)} words from {args.semantle_csv}")
    else:
        words = args.words

    word_sim_matrix = None
    # if args.lambda_sem > 0 and sim_map:
    #     word_sim_matrix = build_word_sim_matrix(words, sim_map)
    #     print(f"[data] Built {len(words)}x{len(words)} similarity matrix.")

    train_semantle_reft(
        model_name=args.model,
        words=words,
        intervention_type=args.intervention_type,
        word_sim_matrix=word_sim_matrix,
        low_rank_dim=args.low_rank_dim,
        layer=args.layer,
        position=args.position,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        beta=args.beta,
        lambda_sem=args.lambda_sem,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
