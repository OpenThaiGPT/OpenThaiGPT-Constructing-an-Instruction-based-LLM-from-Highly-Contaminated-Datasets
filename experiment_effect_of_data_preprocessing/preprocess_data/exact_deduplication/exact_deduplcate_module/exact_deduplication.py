from datasets import Dataset

def hash_text(text: str):
    ord3 = lambda x : '%.3d' % ord(x)
    return int(''.join(map(ord3, text)))

def get_hash(example: Dataset):
    """Get hash of content field."""
    return {"hash": hash(example["text"]), "text_length": len(example["text"])}

def check_uniques(example: Dataset, uniques: dict):
    """Check if the current hash is still in the set of unique hashes based on text length and remove if true."""
    key = example["text_length"]
    # Create a new key if it doesn't exist
    if key not in uniques:
        uniques[key] = set()
    
    # If the hash is still not in the uniques[key], we will add it. We use uniques[key] so that we dont need to iterate through all the uniques.
    if example["hash"] not in uniques[key]:
        uniques[key].add(example["hash"])
        return True
    else:
        return False
    
def exact_deduplicate(dataset: Dataset, splits:list[str], num_processses:int = 0):
    """Exact deduplicate the dataset."""

    for split in splits:
        # Check if the dataset has the text column 
        if "text" not in dataset[split].column_names:
            raise ValueError("The dataset must have the `text` column.")
        else:
            dataset[split] = dataset[split].sort("text")
        dataset[split] = dataset[split].map(get_hash, num_proc=num_processses)
        uniques_hash = {}
        dataset[split] = dataset[split].filter(check_uniques, fn_kwargs={"uniques": uniques_hash}, num_proc=num_processses)
    return dataset