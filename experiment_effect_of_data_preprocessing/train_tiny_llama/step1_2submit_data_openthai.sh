module purge
source /opt/cray/pe/cpe/23.09/restore_lmod_system_defaults.sh
module load Miniconda3
module load cudatoolkit/23.3_11.8
module load PrgEnv-gnu
module load cpe-cuda

conda deactivate
conda activate /project/lt200056-opgpth/new/TinyLlama_2024/.conda_new


SOURCE_DIR=/workspace/source
TOKENIZER_DIR=/workspace/data
OUTPUT_DIR=/workspace/output

python scripts/prepare_openthaigpt.py \
  --source_path $SOURCE_DIR \
  --split train --percentage 1.0 \
  --tokenizer_path $TOKENIZER_DIR \
  --destination_path $OUTPUT_DIR

python scripts/prepare_openthaigpt.py \
  --source_path $SOURCE_DIR \
  --split eval --percentage 1.0 \
  --tokenizer_path $TOKENIZER_DIR \
  --destination_path $OUTPUT_DIR \
  --chunk_size 524544