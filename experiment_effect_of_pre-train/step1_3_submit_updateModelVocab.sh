MODEL_DIR=/workspace/model
TOKENIZER_DIR=/workspace/data
OUTPUT_DIR=/workspace/output



python update_ModelVocab.py \
 --model_name_or_path $MODEL_DIR \
 --tokenizer_name_or_path $TOKENIZER_DIR \
 --output_dir $OUTPUT_DIR