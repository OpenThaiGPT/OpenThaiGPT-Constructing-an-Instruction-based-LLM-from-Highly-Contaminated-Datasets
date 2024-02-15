# Effect of Data preprocessing 

## Step of preprocessing 
1. rule-based_and_perplexity
```
sh /preprocess_data/rule-based_and_perplexity/run_rule-based_and_perplexity.sh
```
2. deduplication
```
sh preprocess_data/deduplication/run_deduplicate.sh
```
3. exact_deduplication
```
sh preprocess_data/exact_deduplication/run_exact_deduplication.sh
```
4. decontamination
```
sh preprocess_data/decontamination/run_decontaminate.sh
```
5. blind_pdpa
```
sh preprocess_data/blind_pdpa/run_blind.sh
```

## Step of pre-train

1. Convert dataset from hf to openthai
```
sh train_tiny_llama/step1_1submit_data_hf_openthai.sh
```
2. Convert the dataset to Tinyllama format
```
sh train_tiny_llama/step1_2submit_data_openthai.sh
```
3. Train Tinyllama models
```
sh train_tiny_llama/step2_submit_train.sh
```
