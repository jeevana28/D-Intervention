# [ICML 2025] Learning Distribution-wise Control in Representation Space for Language Models

This is the official implementation for the paper [Learning Distribution-Wise Control in Representation Space for Language Models].

Generally, we are inspired by the fact that intervention magnitude can be controlled by multiplying a concept vector with a scalar - so why don't we learn that distribution directly? In short, this is  a **deep latent variable model + intervention** research. 

We can directly learn a distribution in latent space for different tasks.

# Requirement

Our codebase is built on pyreft, please install the pyreft from pip:
```bash
pip install pyreft
```
And [`pyvene`](https://github.com/stanfordnlp/pyvene) is the backbone of pyreft library, where serve as great foundation to do intervention research.
# Intervention Type

Pass the intervention class name to `-type` in `train.py`. Available options:

| `-type` value | Description |
|---------------|-------------|
| `LoreftIntervention` | LoReFT (pointwise): h + R^T(Wh + b − Rh) |
| `DistributionalreftIntervention` | **D-ReFT** (distribution-wise): stochastic μ, σ — paper's main method |
| `RedIntervention` | RED: Hadamard rotation + bias |
| `VIBRedIntervention` | VIB + RED |
| `VIBRawreftIntervention` | VIB raw representation |
| `VIBAffinereftIntervention` | VIB affine |
| `VIBLobireftIntervention` | VIB + LobiReFT |
| `LoreftWordIntervention` | LoReFT with per-word b (for Semantle/BO) |
| `NoreftIntervention`, `ConsreftIntervention`, `LobireftIntervention`, `DireftIntervention`, `NodireftIntervention`, `MiniTransformerIntervention` | Other ReFT variants |

# Training Scripts

Generally, if you want to train a distribution-wise intervention on math tasks, run:
```bash
python train.py -task math \
-data_dir dataset \
-model yahma/llama-7b-hf \
-seed 42 \
-l 0 -r 8 -p f7+l7 -e 9 -lr 3e-3 \
-type DistributionalreftIntervention \
-gradient_accumulation_steps 2 \
-batch_size 16 \
-eval_batch_size 4 \
--dropout 0.00 \
--test_split test \
--use_normalized_template \
--share_weights \
--warmup_ratio 0.1 \
--greedy_decoding
```

You can change `DistributionalreftIntervention` to any type above.

---

# How to Run (detailed)

Run all commands from the **D-Intervention** directory so `task_config`, `dataset`, and local `pyreft` are importable:

```bash
cd D-Intervention
```

## 1. Main training: `train.py`

Use `-task`, `-model`, `-type` (intervention), `-l` (layers), `-r` (rank), `-p` (position).

**Math — D-ReFT (paper Appendix B.2):**
```bash
python train.py -task math -data_dir ./datasets -model meta-llama/Llama-3.2-1B \
  -seed 42 -l 0 -r 8 -p f7+l7 -e 9 -lr 3e-3 -type DistributionalreftIntervention \
  -gradient_accumulation_steps 2 -batch_size 16 -eval_batch_size 4 \
  -dropout 0.00 -test_split test -use_normalized_template -share_weights \
  -warmup_ratio 0.1 -greedy_decoding -save_model -output_dir ./official_results
```

**Math — pointwise ReFT (baseline):**
```bash
python train.py -task math -data_dir ./datasets -model meta-llama/Llama-3.2-1B \
  -seed 42 -l 0 -r 8 -p f7+l7 -e 12 -lr 9e-4 -type LoreftIntervention \
  -gradient_accumulation_steps 2 -batch_size 16 -eval_batch_size 4 \
  -dropout 0.00 -test_split test -use_normalized_template -share_weights \
  -warmup_ratio 0.1 -greedy_decoding -save_model -output_dir ./official_results
```

**Commonsense — D-ReFT:**
```bash
python train.py -task commonsense -data_dir ./datasets -model meta-llama/Llama-3.2-1B \
  -seed 42 -l 0 -r 8 -p f7+l7 -e 9 -lr 1e-3 -type DistributionalreftIntervention \
  -gradient_accumulation_steps 2 -batch_size 16 -eval_batch_size 4 \
  -dropout 0.00 -test_split validation -use_normalized_template -share_weights \
  -warmup_ratio 0.1 -greedy_decoding -save_model -output_dir ./official_results
```

**GLUE (e.g. COLA):**
```bash
python train.py -task glue -train_dataset cola -model FacebookAI/roberta-base \
  -seed 42 -l 0 -r 8 -p f3 -e 60 -lr 4e-4 -type LoreftIntervention \
  -batch_size 32 -eval_batch_size 32 -test_split validation -max_length 256 \
  -metric_for_best_model matthews_correlation -dropout 0.2 -warmup_ratio 0.005 \
  -allow_cls_grad -output_dir ./official_results
```

**GSM8K:**
```bash
python train.py -task gsm8k -data_dir ./datasets -model meta-llama/Llama-3.2-1B \
  -seed 42 -l 0 -r 8 -p f7+l7 -e 9 -lr 3e-3 -type DistributionalreftIntervention \
  -gradient_accumulation_steps 2 -batch_size 16 -eval_batch_size 4 \
  -test_split test -use_normalized_template -share_weights \
  -greedy_decoding -save_model -output_dir ./official_results
```

**Key flags:** `-l` = intervention layer(s) (e.g. `0`, `2;10`, `all`); `-r` = rank (8 or 16); `-p` = position (`f7+l7`, `f1+l1`, `all`); `-type` = intervention (see table above); `-save_model` saves under `output_dir/<run_name>`.

## 2. Playground (no training)

```bash
python playground.py
```
Loads model, wraps with ReFT, runs one forward. Edit `MODEL_NAME` in `playground.py` if your model path differs.

## 3. Semantle LoReFT + BO

See `README_SEMANTLE_REFT_BO.md`. Example:
```bash
python semantle_reft_bo.py --mode train_then_bo --model meta-llama/Llama-3.2-1B \
  --words computer laptop keyboard screen mouse --target computer \
  --epochs 2 --n_bo_iter 10 --output_dir ./semantle_reft_bo_out
```

## Data setup

- **Commonsense:** train on Commonsense170K; eval on BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OBQA. Put data under `data_dir` (e.g. `data_dir/commonsense_170k/`, `data_dir/boolq/`) or use HuggingFace dataset names where supported.
- **Math:** train on Math10K; eval on MultiArith, GSM8K, SVAMP, MAWPS, AddSub, AQuA, SingleEq. Same layout under `data_dir`.
- **GLUE:** use `-train_dataset <name>` (e.g. `cola`, `sst2`); data loaded from HuggingFace.
- **GSM8K:** uses last 300 train as validation when `-test_split validation`.

## Paper hyperparameters (Appendix B)

- **ReFT:** lr 9e-4, rank 8 or 16, position f7+l7, batch 8, grad_accum 4, epochs 12.
- **D-ReFT:** lr 1e-3 or 3e-3, rank 8 or 16, position f7+l7, batch 8, grad_accum 4, epochs 9.
- **LoRA:** lr 4e-4, alpha 16, rank 16, position all, epochs 6.
- **RED:** lr 7e-4, position all, epochs 9.

Adjust `-batch_size`, `-gradient_accumulation_steps`, `-lr`, `-e` to match your GPU and reproduce paper results.

## Citation 
```bibtex
@misc{deng2025learningdistributionwisecontrolrepresentation,
      title={Learning Distribution-Wise Control in Representation Space for Language Models}, 
      author={Chunyuan Deng and Ruidi Chang and Hanjie Chen},
      year={2025},
      eprint={2506.06686},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.06686}, 
}
