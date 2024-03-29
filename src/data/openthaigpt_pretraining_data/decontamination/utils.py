from datasets import load_dataset, load_from_disk

import re


def load_data(dataset_arg):
    if dataset_arg.name == "LST20" or dataset_arg.name == "LST20_Test":
        dataset = load_dataset(
            dataset_arg.path_name, dataset_arg.subset, data_dir=dataset_arg.path
        )
    elif dataset_arg.available_on_hub:
        dataset = load_dataset(dataset_arg.path_name, dataset_arg.subset)
    else:
        dataset = load_from_disk(dataset_arg.path_name)
    return dataset


def preprocess_hellaswag(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def generate_query_hellaswag(doc):
    """Modified from  https://github.com/EleutherAI/lm-evaluation-harness/blob/9d06c95346ff393273bbb80379267eef094c0c74/lm_eval/tasks/hellaswag.py"""  # noqa: E501
    ctx = (
        doc["ctx_a_th"] + " " + ""
        if doc["ctx_b_th"] is None
        else doc["ctx_b_th"].capitalize()
    )
    return (preprocess_hellaswag(doc["activity_label_th"] + ": " + ctx),)


def generate_query_xquad(doc):
    """Reference from  https://github.com/EleutherAI/lm-evaluation-harness/blob/4c08d72acdf7c6f5cab5d708e8ef400aea08314c/lm_eval/tasks/squad.py"""  # noqa: E501
    return doc["context"]


def generate_query_thaisum(doc):
    return doc["body"]


def generate_query_multirc_thai(doc):
    return doc["paragraph_TH"]


def generate_query_copa_thai(doc):
    label = doc["label"]
    answer = doc["choice1_th"] if label == 1 else doc["choice2_th"]
    return f'{doc["premise_th"]} f{answer}'


def generate_query_rte_thai(doc):
    return doc["premise"] + " " + doc["hypothesis"]


def generate_query_lst20(doc):
    return "".join(doc["tokens"]).replace("_", " ")


def generate_query_record_thai(doc):
    return doc["passage_TH"]


def generate_query_ted_th_en(doc):
    return doc["translation"]["th"] + " " + doc["translation"]["en"]


MAPPER = {
    "hellaswag_thai": generate_query_hellaswag,
    "xquad": generate_query_xquad,
    "thaisum": generate_query_thaisum,
    "thaisum_test": generate_query_thaisum,
    "multirc_thai": generate_query_multirc_thai,
    "copa_thai": generate_query_copa_thai,
    "rte_thai": generate_query_rte_thai,
    "lst20": generate_query_lst20,
    "lst20_test": generate_query_lst20,
    "record_thai": generate_query_record_thai,
    "ted_talks_iwslt_th_en": generate_query_ted_th_en,
}
