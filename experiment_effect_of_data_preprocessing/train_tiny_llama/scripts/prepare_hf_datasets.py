from typing import Dict, Any, List
import os
import shutil
import json

from datasets import load_from_disk
from tqdm.auto import tqdm

from pathlib import Path

# Define the function to convert the dataset into JSONL chunks
def convert_to_jsonl_chunks(dataset: Dict[str, List[Dict[str, Any]]], 
                            output_dir: str, 
                            chunk_size: int = 1000):
    """
    Convert a dataset into chunks of JSONL files.
    
    Parameters:
    - dataset: Dict containing data splits (e.g., "train", "eval") as keys, 
               and lists of examples as values.
    - output_dir: Directory where the JSONL chunks will be saved.
    - chunk_size: Number of examples per JSONL file.
    """
    if os.path.exists(output_dir):
        # Clear the existing output directory if it exists
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)

    print(dataset, 'dataset')
    
    for split in dataset.keys():
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Divide examples into chunks and save each chunk as a separate JSONL file
        data = dataset[split]
        if split == 'eval':
            selected_chunk_size = 1000
        else:
            selected_chunk_size = chunk_size
        for i in tqdm(range(0, len(data), selected_chunk_size), total = len(data) // selected_chunk_size):
            chunk = data[i:i + selected_chunk_size]
            # print(chunk, 'chunk')
            # break
            chunk_filename = f"{split}_{str(i // selected_chunk_size).zfill(2)}.jsonl"
            chunk_filepath = os.path.join(split_dir, chunk_filename)
            
            with open(chunk_filepath, "w", encoding="utf-8") as f:

                for example_data in zip(*chunk.values()):
                    example = {key: value for key, value in zip(chunk.keys(), example_data)}
                    
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")

def prepare(
        input_path: Path,
        output_path: Path
    ) -> None:
    hf_datasets = load_from_disk(input_path)
    print('loaded dataset')

    convert_to_jsonl_chunks(hf_datasets, output_path, 10_000)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)

