python tools/preprocess_data.py \
       --input ~/datasets/megatron_slim/example_train_0.jsonl \
       --output-prefix ~/datasets/megatron_slim/out \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model ~/model_weights/Llama-2-7b-hf/tokenizer.model \
       --dataset-impl mmap \
       --append-eod \
       --workers 8 \
