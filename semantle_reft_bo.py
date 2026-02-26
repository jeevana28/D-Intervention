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
    TrainingArguments,
)
from pyreft.reft_trainer import ReftTrainerForCausalLM

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


def build_word_sim_matrix(
    words: List[str], sim_map: Dict[str, float]
) -> torch.Tensor:
    """
    N x N target similarity matrix for the semantic structure loss.
    sim_matrix[i][j] = sim[i] * sim[j]  (both measured vs. the target word).
    """
    sims = torch.tensor([sim_map.get(w, 0.0) for w in words], dtype=torch.float32)
    mat = sims.unsqueeze(1) * sims.unsqueeze(0)
    mat.fill_diagonal_(1.0)
    return mat


def build_semantle_data(
    words: List[str],
    use_word_ids: bool = False,
    b_vectors: Optional[Dict[str, np.ndarray]] = None,
    low_rank_dim: int = 8,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Build {word, subspaces} dicts for LoReftSemantleDataset.
    use_word_ids=True  -> subspaces = [word_id]  (DistributionalWordIntervention)
    use_word_ids=False -> subspaces = float list  (LoreftWordIntervention)
    """
    word_to_id = {w: i for i, w in enumerate(words)}
    data = []
    for w in words:
        item = {"word": w}
        if use_word_ids:
            item["subspaces"] = [word_to_id[w]]
        else:
            item["subspaces"] = (b_vectors[w].astype(np.float32).tolist()
                                 if b_vectors and w in b_vectors
                                 else [0.0] * low_rank_dim)
        data.append(item)
    return data, word_to_id


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
    beta: float = 0.5,
    lambda_sem: float = 0.0,
    output_dir: str = "./semantle_reft_out",
    cache_dir: Optional[str] = None,
    seed: int = 42,
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
    use_word_ids = (intervention_type == "DistributionalWordIntervention")
    iv_cls = INTERVENTION_TYPES[intervention_type]
    iv_kwargs = dict(
        embed_dim=hidden_size,
        low_rank_dimension=low_rank_dim,
        dropout=0.0,
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device=DEVICE,
    )
    if use_word_ids:
        iv_kwargs["num_words"] = len(words)
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
        words, use_word_ids=use_word_ids,
        b_vectors=None if use_word_ids else b_vectors,
        low_rank_dim=low_rank_dim,
    )
    from datasets import Dataset as HFDataset
    train_dataset = LoReftSemantleDataset(
        task="semantle", data_path="semantle", tokenizer=tokenizer,
        data_split="train", dataset=HFDataset.from_list(data),
        seed=seed, max_n_example=len(data),
        num_interventions=1, position=position,
        share_weights=False, low_rank_dimension=low_rank_dim,
    )

    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, learning_rate=lr,
        logging_steps=10, save_strategy="no",
        remove_unused_columns=False, report_to="none",
    )
    trainer = ReftTrainerForCausalLM(
        model=reft_model, tokenizer=tokenizer, args=training_args,
        train_dataset=train_dataset,
        data_collator=SemantleDataCollator(tokenizer=tokenizer, model=model),
        word_sim_matrix=word_sim_matrix,
        lambda_sem=lambda_sem,
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
    parser.add_argument("--beta", type=float, default=0.5,
                        help="KL weight for DistributionalWordIntervention.")
    parser.add_argument("--lambda_sem", type=float, default=0.0,
                        help="Semantic structure loss weight.")
    parser.add_argument("--output_dir", type=str, default="./semantle_reft_out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    sim_map: Dict[str, float] = {}
    if args.semantle_csv:
        _, words, sim_map = load_semantle_csv(args.semantle_csv, top_k=args.top_k)
        print(f"[data] {len(words)} words from {args.semantle_csv}")
    else:
        words = args.words

    word_sim_matrix = None
    if args.lambda_sem > 0 and sim_map:
        word_sim_matrix = build_word_sim_matrix(words, sim_map)
        print(f"[data] Built {len(words)}x{len(words)} similarity matrix.")

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
        beta=args.beta,
        lambda_sem=args.lambda_sem,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
