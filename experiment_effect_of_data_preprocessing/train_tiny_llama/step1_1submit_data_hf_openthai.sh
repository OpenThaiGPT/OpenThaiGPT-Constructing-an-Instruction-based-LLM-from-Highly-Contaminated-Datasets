HUGGINGFACE_DATASET_SRC_PATH=/workspace/datset1
JSONL_DATASET_SAVE_PATH=/workspace/datset2


python scripts/prepare_hf_datasets.py \
   $HUGGINGFACE_DATASET_SRC_PATH \
   $JSONL_DATASET_SAVE_PATH