import pyvene as pv
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollator,
    DataCollatorForSeq2Seq,
    AutoTokenizer
)
from transformers.trainer_utils import (
    EvalPrediction,
    has_length,
    denumpify_detensorize
)
from datasets import Dataset
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from tqdm import tqdm
import os
import torch
import re
import evaluate
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import logging
from .interventions import VIBreftIntervention, VIBRawreftIntervention, DistributionalWordIntervention

logger = logging.get_logger(__name__)

@dataclass
class ReftDataCollator(object):
    """Collate examples for ReFT."""

    data_collator: DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_inputs = self.data_collator(instances)
        max_seq_length = batch_inputs["input_ids"].shape[-1]
        batch_inputs["intervention_locations"] = batch_inputs["intervention_locations"][..., :max_seq_length]
        return batch_inputs


def make_data_collator(tokenizer, model) -> ReftDataCollator:
    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
        max_length=2048,
    )
    return ReftDataCollator(data_collator=data_collator_fn)


def make_dataloader(dataset: Dataset, batch_size: int, collate_fn: DataCollatorForSeq2Seq, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)


class ReftTrainer(Trainer):
    def __init__(self, *args, word_sim_matrix=None, lambda_sem: float = 0.0, **kwargs):
        # transformers ≥4.47 renamed `tokenizer` → `processing_class`
        if "tokenizer" in kwargs:
            import transformers as _tv
            _major, _minor = (int(x) for x in _tv.__version__.split(".")[:2])
            if (_major, _minor) >= (4, 47):
                kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(*args, **kwargs)
        # Optional (N, N) float32 tensor of pairwise semantic similarities.
        # Enables the semantic structure loss: similar words → nearby mu vectors.
        self.word_sim_matrix = word_sim_matrix
        self.lambda_sem = lambda_sem

    def save_model(self, output_dir, _internal_call=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_intervention(
            save_directory=f"{output_dir}/intervenable_model", 
            include_model=True
        )

    def _load_best_model(self):
        logger.warning(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        self.model.load_intervention(
            f"{self.state.best_model_checkpoint}/intervenable_model", 
            include_model=True
        )

    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        # run intervened forward pass
        unit_locations = None
        if "intervention_locations" in inputs:
            if inputs["intervention_locations"].dim() == 3:
                unit_locations={"sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist()
                )}
            else:
                # this is dummy for lora only baseline
                unit_locations={"sources->base": (None, 0)}
        base_outputs, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None
        )
        
        # pyvene may store each intervention as a (obj, ...) tuple or as the object itself
        def _get_intervention(v):
            return v[0] if isinstance(v, (list, tuple)) else v

        first_intervention = _get_intervention(list(intervenable.interventions.values())[0])

        output = cf_outputs if cf_outputs is not None else base_outputs
        total_loss = output.loss

        # ── KL divergence loss ───────────────────────────────────────────────
        # For VIBreftIntervention / VIBRawreftIntervention: opt-in via use_compression=True.
        # For DistributionalWordIntervention: applied whenever beta > 0 (no flag needed).
        kl_loss = torch.tensor(0.0, device=total_loss.device)
        _has_kl = isinstance(first_intervention, (VIBreftIntervention, VIBRawreftIntervention)) and getattr(first_intervention, 'use_compression', False)
        _has_kl = _has_kl or (
            isinstance(first_intervention, DistributionalWordIntervention)
            and getattr(first_intervention, 'beta', 0.0) > 0
        )
        if _has_kl:
            beta = first_intervention.beta
            for intervention_val in intervenable.interventions.values():
                iv = _get_intervention(intervention_val)
                if hasattr(iv, 'last_mu') and hasattr(iv, 'last_logvar'):
                    kl_loss = kl_loss + iv.compute_kl_divergence(iv.last_mu, iv.last_logvar)
            total_loss = total_loss + beta * kl_loss

        # ── Semantic structure loss ──────────────────────────────────────────
        # Encourage mu vectors of semantically similar words to be nearby in
        # b-space so that interpolating b vectors yields semantically
        # intermediate words.
        
        # Applied whenever lambda_sem > 0 for DistributionalWordIntervention,
        # regardless of use_compression.  Crucially, the loss is computed over
        # ALL word pairs (full embedding table), not just the in-batch subset,
        # so every word's position in b-space is shaped at every gradient step.
        # sem_loss = torch.tensor(0.0, device=total_loss.device)
        # if (
        #     self.word_sim_matrix is not None
        #     and self.lambda_sem > 0
        #     and isinstance(first_intervention, DistributionalWordIntervention)
        #     and hasattr(first_intervention, 'word_mu')
        # ):
        #     # Use the full embedding table so all N*(N-1)/2 word-pair
        #     # constraints are active at every step (not just C(batch,2) pairs).
        #     all_mu = first_intervention.word_mu.weight.float()   # (N, D)
        #     mu_norm = F.normalize(all_mu, dim=-1)
        #     cos_sim_b = torch.mm(mu_norm, mu_norm.T)              # (N, N)

        #     target_sim = self.word_sim_matrix.to(all_mu.device)   # (N, N)
        #     sem_loss = F.mse_loss(cos_sim_b, target_sim)
        #     total_loss = total_loss + self.lambda_sem * sem_loss

        _is_distributional_word = isinstance(first_intervention, DistributionalWordIntervention)
        if _has_kl or (_is_distributional_word and self.lambda_sem > 0):
            self.log({
                "ce_loss":    output.loss.item(),
                "kl_loss":    kl_loss.item(),
                "total_loss": total_loss.item(),
            })

        return (output, output) if return_outputs else total_loss


class ReftTrainerForCausalLM(ReftTrainer):
    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True)


class ReftTrainerForSequenceClassification(ReftTrainer):
    def evaluate(
        self, ignore_keys,
    ):

        # ensure everything is in eval mode
        self.model.model.eval()
        for k,v in  self.model.interventions.items():
            _ = v[0].eval()
        
        batch_size = self.args.eval_batch_size
        data_collator = self.data_collator
        eval_dataset = self.eval_dataset
        intervenable = self.model
        
        dataloader = make_dataloader(
            eval_dataset, batch_size, data_collator, shuffle=False)

        logger.info(f"***** Running In-Training Evaluation *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        eval_iterator = tqdm(dataloader, position=0, leave=True)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for step, inputs in enumerate(eval_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.model.get_device())
                
                # [layers, batch_size, positions]
                intervention_locations = inputs["intervention_locations"].permute(1, 0, 2).tolist()
                _, cf_outputs = intervenable(
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                    unit_locations={"sources->base": (None, intervention_locations)})
            
                all_preds += [cf_outputs.logits]
                all_labels += [inputs["labels"]]
        all_preds = torch.cat(all_preds, dim=0).cpu().to(torch.float32)
        all_labels = torch.cat(all_labels, dim=0).cpu().to(torch.float32)
        metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        metrics = denumpify_detensorize(metrics)
        
        metric_key_prefix = "eval"
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        
        return metrics
        
