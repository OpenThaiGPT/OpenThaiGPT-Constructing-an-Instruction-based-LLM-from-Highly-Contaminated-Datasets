MODEL_DIR=/workspace/model
SP_DIR=/workspace/data
OUTPUT_DIR=/workspace/output

python ../src/model/scripts/llama_thai_tokenizer/merge_tokenizer.py \
    --llama_path $MODEL_DIR \
    --sp_path $SP_DIR \
    --output_path $OUTPUT_DIR