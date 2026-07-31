#!/usr/bin/env bash
# Inkling-Small (276B) LoRA GRPO (r=32, alpha=32, all-linear) on 4 nodes x
# 8 H200 — the validated profile: same parallel layout as full-parameter,
# ctx 4096 / response 2048, rollout 64 prompts x 8 samples (gbs 128), and
# lr 2e-4.  Measured: reward +0.011/rollout.
#
# The lr is deliberately ~30x the full-parameter one: LoRA's B factors start
# at zero, so |delta-W| = |B@A| accumulates at ~lr x steps x |A|; at a
# full-parameter-style lr the adapter needs hundreds of rollouts before the
# policy visibly moves.  No optimizer offload: the adapter states are tiny.
set -euxo pipefail
cd "$(dirname "$0")/.."

export MILES_SCRIPT_EXTERNAL_RAY=1
export MASTER_ADDR=${MASTER_ADDR:?set to the head node IP}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}

MODEL_DIR=${MODEL_DIR:-/root/models}
HF_CKPT=${HF_CKPT:-${MODEL_DIR}/Inkling-Small}
TORCH_DIST=${TORCH_DIST:-${MODEL_DIR}/Inkling-Small_torch_dist}

python3 scripts/run_inkling.py train \
  --model-name Inkling-Small --train-mode lora --task dapo_math \
  --num-nodes 4 --num-gpus-per-node 8 \
  --hf-checkpoint "$HF_CKPT" \
  --torch-dist "$TORCH_DIST" \
  --data-dir "${DATA_DIR:-/root/datasets}" \
  --lr 2e-4 \
  --rollout-batch-size 64 --global-batch-size 128 \
  --sglang-context-length 4096 --rollout-max-response-len 2048 \
  --megatron-path "${MEGATRON_PATH:-/root/Megatron-LM}" \
  "$@"
