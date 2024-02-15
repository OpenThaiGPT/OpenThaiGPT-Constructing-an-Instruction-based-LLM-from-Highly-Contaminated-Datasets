## Train DPO

```bash
002-001-dpo-temp-0_3-v-all-ref.sh
```

### Configuration

- BASE_MODEL: Name of Model for save.
- DATA_PATH: Dataset Path.
- EPOCH: Num Train Epoch.
- LR: 2e-5 for full finetune and 2e-4 for lora.
- GRADIENT_ACCUMULATION_STEPS: Accumulation step.
- MAX_LEN: Max training length.
- MAX_PROMPT_LEN: Max training prompt length.
- MICRO_BSZ: Batch size per step.
- VAL_SIZE: Split validation set.
- WANDB_NAME: Wandb project name.
- WARMUP_STEPS: Warmup step for scheduler.
