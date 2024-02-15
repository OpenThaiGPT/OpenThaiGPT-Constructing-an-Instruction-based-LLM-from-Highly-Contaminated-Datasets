import sys
import os.path as op 

from exact_deduplcate_module.exact_deduplication import exact_deduplicate
import hydra
from datasets import load_from_disk

# Exact deduplcation
@hydra.main(version_base=None, config_path="./config", config_name="exact_deduplicate_config")
def exact_deduplicate_pipeline(cfg):
    dataset = load_from_disk(cfg.dataset.path)
    exact_deduplcated_dataset = exact_deduplicate(dataset, cfg.dataset.splits, cfg.global_config.num_process)
    exact_deduplcated_dataset.save_to_disk(cfg.dataset.save_path)

if __name__ == "__main__":
    exact_deduplicate_pipeline()
