#!/bin/bash
python semantle_reft_bo.py \
  --mode train \
  --semantle_csv /work/pi_mccallum_umass_edu/jkarnuthala_umass_edu/BOPRO-ICLR-2025/data/semantle/train/computer.csv \
  --top_k 50 \
  --intervention_type DistributionalWordIntervention \
  --epochs 30 \
  --lambda_sem 0.0 \
  --beta 0.5 \
  --output_dir out_distributional \
  --cache_dir /datasets/ai/llama3/hub \
  --wandb_project semantle-reft \
  --wandb_run_name distributional-computer
