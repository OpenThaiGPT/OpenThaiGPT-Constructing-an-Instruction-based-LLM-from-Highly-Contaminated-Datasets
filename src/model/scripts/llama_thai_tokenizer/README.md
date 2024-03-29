# OpenThaiGPT Merged Tokenizer Pipeline

We plan to use [Pretrained LLaMA Model](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) as a base model for finetuning but LLaMA Tokenizer (BPE Tokenizer) have less Thai vocabulary, this pipeline intend to extent vocabulary of LlaMA tokenizer, using merge method following [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/README_EN.md)

## Method

- merge LLaMA tokenizer by extenting vocabulary and merge rule from BPE Thai tokenizer
  ![method](merge_method.png)

## merge and save merge tokenizer

1. load LLaMA tokenizer by pass your model name to an argument

   ```bash
   python load_tokenizer.py --model_name meta-llama/Llama-2-7b --output_path <output_tokenizer_path>
   ```

   argument

   - model_name: name of LLaMA model (huggingface)
   - output_path: path to save tokenizer

2. prepare Thai BPE Tokenizer

   - if you don't have your Thai BPE Tokenizer, you need to train following [spm_training/README.md](../spm_training/README.md)

3. merge tokenizer by running following script

   - must have Thai BPE tokenizer, don't forget set <thai_sp_path> to path of Thai BPE Tokenizer

   ```bash
   python merge_tokenizer.py --llama_path <llama_model_path> --thai_sp_path <spm_model_path> --output_path <output_tokenizer_path>
   ```

   argument

   - llama_path: path to LLaMA tokenizer huggingface or local
   - thai_sp_path: path to Thai BPE tokenizer on local
   - output_path: path to save tokenizer

## To test merged tokenizer

1.  run llama_thai_token_test.py and inference time checked

    ```bash
      python time_inference_check.py --llama_path <llama_model_path> --thai_sp_path <spm_model_path>
    ```

    - llama_path: path to LLaMA tokenizer huggingface or local
    - thai_sp_path: path to Thai BPE tokenizer on local

### Results after merging

| Text                                                         | LLaMA Tokenizer                                                                                                                                                                                                                                                                                                                                         | Merged Tokenizer                                                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| การใช้งานหลักของ LLaMA คือการวิจัยเกี่ยวกับรูปแบบภาษาที่ใหญ่ | ['▁', 'ก', 'า', 'ร', '<0xE0>', '<0xB9>', '<0x83>', 'ช', '้', 'ง', 'า', 'น', 'ห', 'ล', 'ั', 'ก', 'ข', 'อ', 'ง', '▁L', 'La', 'MA', '▁', 'ค', 'ื', 'อ', 'ก', 'า', 'ร', 'ว', 'ิ', 'จ', 'ั', 'ย', 'เ', 'ก', 'ี', '่', 'ย', 'ว', 'ก', 'ั', 'บ', 'ร', 'ู', 'ป', 'แ', 'บ', 'บ', 'ภ', 'า', 'ษ', 'า', 'ท', 'ี', '่', '<0xE0>', '<0xB9>', '<0x83>', 'ห', 'ญ', '่'] | ['▁การใช้งาน', 'หลักของ', '▁L', 'La', 'MA', '▁คือการ', 'วิจัย', 'เกี่ยวกับ', 'รูปแบบ', 'ภาษา', 'ที่ใหญ่'] |
| LLaMAมุ่งเน้นที่การศึกษารูปแบบภาษาที่กว้างขวาง               | ['▁L', 'La', 'MA', '▁', 'ม', 'ุ', '่', 'ง', 'เ', 'น', '้', 'น', 'ท', 'ี', '่', 'ก', 'า', 'ร', 'ศ', '<0xE0>', '<0xB8>', '<0xB6>', 'ก', 'ษ', 'า', 'ร', 'ู', 'ป', 'แ', 'บ', 'บ', 'ภ', 'า', 'ษ', 'า', 'ท', 'ี', '่', 'ก', 'ว', '้', 'า', 'ง', 'ข', 'ว', 'า', 'ง']​                                                                                          | ['▁L', 'La', 'MA', '▁มุ่ง', 'เน้น', 'ที่การ', 'ศึกษา', 'รูปแบบ', 'ภาษา', 'ที่', 'กว้างขวาง']              |
| ขอเพิ่มสัก1pointก็ยังดีครับ                                  | ['▁', 'ข', 'อ', 'เ', 'พ', 'ิ', '่', 'ม', 'ส', 'ั', 'ก', '▁', '1', '▁point', '▁', 'ก', '็', 'ย', 'ั', 'ง', 'ด', 'ี', 'ค', 'ร', 'ั', 'บ']                                                                                                                                                                                                                 | ['▁ขอ', 'เพิ่ม', 'สัก', '▁1', '▁point', '▁ก็ยัง', 'ดีครับ']                                               |
| Convert Pretrained LLaMa to Support Thai Token               | ['▁Convert', '▁P', 'ret', 'rained', '▁L', 'La', 'Ma', '▁to', '▁Support', '▁Thai', '▁Token']                                                                                                                                                                                                                                                             | ['▁Convert', '▁P', 'ret', 'rained', '▁L', 'La', 'Ma', '▁to', '▁Support', '▁Thai', '▁Token']               |

- EngOnly time: 0.00027060508728027344
- EngThai time: 0.00016260147094726562
- time diff: 0.00010800361633300781
