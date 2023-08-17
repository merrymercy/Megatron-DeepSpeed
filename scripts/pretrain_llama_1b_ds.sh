#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8

MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=~/tmp
DATA_PATH=~/datasets/megatron_slim/out_text_document
TOKENIZER_PATH=~/model_weights/Llama-2-7b-hf/tokenizer.model

DS_CONFIG=deepspeed.json
GLOBAL_BATCH_SIZE=512
MICRO_BATCH_SIZE=16
ZERO_STAGE=2

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  }
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

MODEL_ARGS="
    --num-layers 20 \
    --hidden-size 2048 \
    --ffn-hidden-size 5376 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
"

HP_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr 2e-4 \
    --train-iters 20000 \
    --lr-decay-iters 20000 \
    --lr-decay-style cosine \
    --min-lr 2e-5 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction .01 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0 \
    --bf16 \
    --no-query-key-layer-scaling \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --normalization rmsnorm \
    --disable-bias-linear \
    --use-flash-attn \
    --use-distributed-optimizer \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 100,0,0 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model $TOKENIZER_PATH \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 5
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $MODEL_ARGS \
    $HP_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    $ds_args
