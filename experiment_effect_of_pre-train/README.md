# Experiment Effect of pretrained

This experiment aims to measure how the effectiveness of pretrained LLM in Thai language

Experiment Models:
- Llama-v2-7B
- Mistral-7B
- SeaLion-7B
- Qwen-7B

Experiment Setups
1. Preparing Tokenizer
we training additional Thai langue tokenizers
note that we skipped SeaLion-7B since there are large amounts containing Thai tokens
1.1 Training SPM tokenizer is required for constructing LLM
at config file: `src/model/configuration_example/spm/training_v1.yaml`
```
output_path: path/to/output
vocab_size: 30000
is_slurm: false
load_dataset_path: null
load_dataset_name: dataset_name
load_dataset_local_path: path/to/dataset
load_dataset_data_type: null
large_corpus: true
mode: bpe # set spm if Llama-v2-7B, Mistral-7B, and bpe for Qwen-7B
```
run
```python
python src/model/scripts/spm_training/train.py
```

1.2. Run merging script for remote tokenizers and newly trained tokenizers
run
```python
python src/model/scripts/llama_thai_tokenizer/merge_tokenizer.py \
    --llama_path  path/to/remote_tokenizer \
    --sp_path  path/to/new_tokenizers.model \
    --output_path path/to/output
```

1.3. Now we update the vocab size
run
```python
python src/model/scripts/llama_thai_tokenizer/merge_tokenizer.py \
    --llama_path  path/to/remote_tokenizer \
    --sp_path  path/to/new_tokenizers.model \
    --output_path path/to/output
```

2. we preprocess configs for later continued pretraining in step 3
there are 3 config files to add: dataset, model, data_process
you can look for an example format inside a file in the same folder
at config file: `src/model/configuration_example/dataset/dataset.yaml`
change here, denote `#` as comment:
```yaml
...
tokenizer:
  pretrained_model_name_or_path path/to/model # from step 1.3
  tokenizer_class: LlamaTokenizer # if use spm as previous else AutoTokenizer
...
```
at config file: `src/model/configuration_example/model/model.yaml`
change here:
```yaml
...
train:
  dataset_name: path/to/dataset
  split: train
  from_disk: True
eval:
  dataset_name: path/to/dataset
  split: eval
  from_disk: True
...
```
at config file: `src/model/configuration_example/data_process/data_process.yaml`
change here:
```yaml
...
max_tokens: 2048
save_path: path/to/save
...
```
run preprocess script
```python
python src/model/scripts/lighting_training/data_preprocessing.py
```
3. we can submit the training script at
Some parameters concerns, editing in `step3_smultinode_deepspeed.sh`

| Parameter                                                    | Explanation                                                                                                                                                                    |
|--------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model_name_or_path`:                                        | path of model from previous                                                                                                                                                     |
| `tokenizer_name_or_path`                                     |  path of tokenizer from   previous                                                                                                                                              |
| `data_path`                                                  | data path from step 2                                                                                                                                                           |
| `data_weight`                                                | ratio of th en (over 1.0 meaning oversampling) depends on the   smallest size first Exp 1: th=0.9 en=0.1, Exp3 (th,en) is (.9,.1), (.85,.15),   (.8,.2), (.75,.25) respectively |
| `per_device_train_batch_size`   `per_device_eval_batch_size` | batch size needs to be adjusted according to GPU RAM                                                                                                                            |
| `gradient_accumulation_steps`                                | for compensate token w/ formula Total tokens =   n_gpus(n_nodes=4) * max_len=2048 * per_device_train_batch_size *   gradient_accumulation_steps                                 |

after that, we can run `step3_submit_multinode_deepspeed.sh`
