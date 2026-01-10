# TO DO
# - parse complete dataset
# - look at labels used in chia annotation -- decide if some should be merged or removed
# - develop NER training/evaluation pipeline





# PARSING THE CHIA DATASET FOR NER TASKS - SENTENCE-WISE APPROACH

# %%
# imports
import os
import glob
import re
import json
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# %%
# function definitions

# UNCHANGED FROM CRITERIA-WISE PARSING VERSION
def parse_brat_file(ann_path):
    """
    Parses an .ann file to extract named entities with global character offsets.

    This function handles both continuous and discontinuous spans (e.g., "1311 1318;1334 1341").
    Discontinuous spans are flattened into multiple distinct entity mentions sharing the same ID,
    as standard BERT NER cannot represent non-contiguous entities.

    Args:
        ann_path (str): The full file path to the .ann annotation file.

    Returns:
        list[dict]: A sorted list of entity dictionaries. Each dictionary contains:
            - "id" (str): The entity ID (e.g., "T1").
            - "type" (str): The entity label (e.g., "Condition").
            - "start" (int): The starting character offset relative to the file start.
            - "end" (int): The ending character offset relative to the file start.
            - "text" (str): The text content of the entity (if available).
    """

    entities = list()
    
    try:
        with open(ann_path, "r", encoding="UTF-8") as f:
            for line in f:
                line = line.strip()
                # focus on "T"-annotations for named entities
                if line.startswith("T"):
                    parts = line.split("\t")
                    # ignore malformed annotations 
                    if len(parts) < 2: continue
                    
                    id_ = parts[0]
                    type_and_offsets = parts[1]
                    text = parts[2] if len(parts) > 2 else ""
                    
                    # parse type and offsets
                    # example: "Condition 1311 1318; 1334 1341" (discontinuous entity)
                    meta_parts = type_and_offsets.split(" ")
                    entity_type = meta_parts[0] # example: "Condition"
                    offset_string = " ".join(meta_parts[1:]) # example: "1311 1318; 1334 1341"
                     
                    # regex to capture all start/end pairs
                    # handling discontinuous spans by treating fragments as separate mentions
                    raw_spans = re.findall(r"(\d+)\s(\d+)", offset_string) # example: [("1311", "1318"), ("1334", "1341")]
                    
                    for start, end in raw_spans:
                        entities.append({
                            "id": id_,
                            "type": entity_type,
                            "start": int(start),
                            "end": int(end),
                            "text": text
                        })
                        
    except FileNotFoundError:
        print(f"Warning: File not found: {ann_path}")
    
    
    # sort entities by start position for correct processing order
    data = sorted(entities, key=lambda x: x["start"])
    
    return data                   
            


def split_text_and_realign_entities(text, global_entities):
    """
    Splits the full document text into lines and recalculates entity indices 
    relative to the start of each line. Crucial for sentence-wise processing.

    Args:
        text (str): The full content of the .txt file.
        global_entities (list[dict]): List of entities with global offsets (from parse_brat_file).

    Returns:
        data_linelevel (list[dict]): A list of line-level samples. Each dict contains:
            - "text" (str): The text of the specific line.
            - "entities" (list[dict]): Entities within this line, with 'start'/'end' 
              recalculated to be relative to the line's start (0-indexed).
    """
    
    linelevel_data = list()
    current_global_idx = 0
    
    # newlines as criterion boundaries in chia dataset
    lines = text.split("\n")
    
    for line in lines:
        line_len = len(line)
        line_end_idx = current_global_idx + line_len
        
        local_entities = list()
        
        # find entities within current line
        for ent in global_entities:
            # check strict containment:
            # entity start/end within line boundaries
            if ent["start"] >= current_global_idx and ent["end"] <= line_end_idx:
                # line-related indices for entity
                new_start = ent["start"] - current_global_idx
                new_end = ent["end"] - current_global_idx

                local_entities.append({
                    "id": ent["id"],
                    "type": ent["type"],
                    "start": new_start,
                    "end": new_end,
                    "text": ent["text"]
                })
        
        
        # ignore empty lines, add non-empty lines
        if line.strip():
            linelevel_data.append({
                "text": line,
                "entities": local_entities
            })
        
        # update global index for next line
        # +1 for newline character
        current_global_idx += line_len + 1
        
    return linelevel_data
            


def process_file_line_by_line(txt_path, ann_path, tokenizer, label2id):
    """
    Processing of a pair of .txt-file and .ann-file into tokens and aligned labels.
    - read text, parse entities
    - split text into lines, realign entity indices
    - tokenize each line, align character labels to BERT subword tokens (BIO scheme)

    Args:
        txt_path (str): Path to the .txt file
        ann_path (str): Path to the .ann file
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer
        label2id (dict): label mapping to ids, updated in-place4

    Returns:
        list[dict]: A list of processed samples ready for the HuggingFace dataset
                    Each dict contains 'input_ids', 'attention_mask', 'labels', etc.
    """
    
    # open .txt-file
    with open(txt_path, "r", encoding="UTF-8") as f:
        text = f.read()
    
    # prase text and annotations into line-level samples
    global_entities = parse_brat_file(ann_path)
    linelevel_data = split_text_and_realign_entities(text, global_entities)
    
    # process each line separately
    processed_data = list()
    for item in linelevel_data:
        line_text = item["text"]
        line_entities = item["entities"]
        
        # tokenization
        line_tokenized = tokenizer(
            line_text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_offsets_mapping=True
        )
        index_mapping = line_tokenized["offset_mapping"]
        
        # initialize labels to "O"
        # for each token in line
        labels = [label2id["O"]] * len(line_tokenized["input_ids"])

        # align entity indices with token-level BIO labels
        for ent in line_entities:
            entity_type = ent["type"]
            b_label = f"B-{entity_type}"
            i_label = f"I-{entity_type}"
            
            # update label2id mapping dynamically
            if b_label not in label2id: label2id[b_label] = len(label2id)
            if i_label not in label2id: label2id[i_label] = len(label2id)
            
            # retrieve label ids
            b_id = label2id[b_label]
            i_id = label2id[i_label]
            
            found_start = False
            for idx, (start, end) in enumerate(index_mapping):
                # skip special tokens ([CLS], [PAD], etc.)
                if start == 0 and end == 0:
                    continue
                
                # overlap between token and entity-id
                if start >= ent["start"] and end <= ent["end"]:
                    # beginning of entity
                    if not found_start:
                        labels[idx] = b_id
                        found_start = True
                    # inside of entity
                    else:
                        labels[idx] = i_id
                        
                # edge case: entity-ids include token boundaries
                # treat as beginning of entity
                elif start < ent["start"] and end > ent["start"]:
                    labels[idx] = b_id
                    found_start = True

        # collect processed line dictionary in list of processed lines
        processed_data.append({
            "input_ids": line_tokenized["input_ids"],
            "attention_mask": line_tokenized["attention_mask"],
            "labels": labels,
            "file_name": os.path.basename(txt_path),
            "sentence_text": line_text
            })
    
    return processed_data




def load_chia_dataset(data_dir, model_checkpoint):
    """
    Main driver function: load and process files of chia dataset.
    Returns a formatted chia-dataset for NER.
    
    Args:
        data_dir (str): Directory containing .txt and .ann files from chia dataset
        model_checkpoint (str): BERT-based model checkpoint from HuggingFace, used as tokenizer
        
    Returns:
        tuple: (hf_dataset, label2id, id2label) - processed dataset with mapping dictionaries
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # initialize label map with "O" label
    label2id = {"O": 0}
    data_lines = list()
    
    # retrieve all .txt files in data_dir
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    print(f"Processing {len(txt_files)} files from {data_dir}...")
    
    # open, parse, and process each file pair
    for txt_path in txt_files:
        ann_path = txt_path.replace(".txt", ".ann")
        
        # no .ann-file corresponding to .txt-file
        # -> skip .txt-file
        if not os.path.exists(ann_path):
            continue
        
        # retrieve criteria type from filename
        # default to "unknown"
        file_name = os.path.basename(txt_path)
        criteria_type = "unknown"
        if "_exc" in file_name:
            criteria_type = "exclusion"
        elif "_inc" in file_name:
            criteria_type = "inclusion"
            
        
        # process file-pair line by line
        # returns a list of line-level processed data
        # label2id updated in-place - no need to return
        processed_lines = process_file_line_by_line(txt_path, ann_path, tokenizer, label2id)
        
        # add criteria type to each processed line
        for line in processed_lines:
            line["criteria_type"] = criteria_type
            data_lines.append(line)
        
    # convert lsit of dicts to HuggingFace dataset
    # via Pandas DataFrame for convenience
    df = pd.DataFrame(data_lines)
    dataset = Dataset.from_pandas(df)
        
    # convert "criteria_type" to ClassLabel for stratification compatibility
    features = dataset.features.copy()
    features["criteria_type"] = ClassLabel(names=["exclusion", "inclusion", "unknown"])
    dataset = dataset.cast_column("criteria_type", features["criteria_type"])
    
    # create id2label mapping from label2id
    id2label = {v: k for k, v in label2id.items()}
    
    print(f"Total line-level samples processed: {len(dataset)}")
    
    return dataset, label2id, id2label



def save_label_map(label2id, id2label, output_dir, filename="label_map.json"):
    """
    Saves label mappings to a JSON file.

    Args:
        label2id (dict): Mapping of Label -> ID.
        id2label (dict): Mapping of ID -> Label.
        output_dir (str): Directory to save json-file
        filename (str): Name of the output file
            
    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    path = os.path.join(output_dir, filename) 
    data = {
        "label2id": label2id,
        "id2label": id2label
    }
    
    try:
        with open(path, "w", encoding="UTF-8") as f:
            json.dump(data, f, indent=4)
        print(f"Label mappings saved to {path}")
    except Exception as e:
        print(f"Failed to save label mappings: {e}")

    return None




def get_entity_presence_matric(dataset, label_column="labels"):
    """
    Converts a dataset of NER sequences into a binary presence matrix 
    for iterative stratification.
    
    Args:
        dataset: HuggingFace dataset
        label_column: Name of the column containing NER-label ids
        
    Returns:
        np.array: A binary matrix of shape (n_samples, n_entity_types)
                  where 1 indicates the entity type is present in the sentence.
    """
    
    # identify unique labels in dataset
    # excluding "O" (non-entity label)
    all_labels = list()
    for sent_labels in dataset["labels"]:
        all_labels.extend(sent_labels)
    unique_labels = set(all_labels)
    unique_labels.remove(0)
    
    # map label-ids to matrix columns
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    # build matrix
    # rows = n sentences
    # columns = n entity types (labels)
    y_matrix = np.zeros((len(dataset), len(unique_labels)))
    for row_idx, labels in enumerate(dataset[label_column]):
        sent_labels = set(label for label in labels if label != 0)
        for label in sent_labels:
            if label in label_to_idx:
                col_idx = label_to_idx[label]
                y_matrix[row_idx, col_idx] = 1
    
    return y_matrix
    


# ITERATIVE STRATIFICATION BY LABELS
def split_and_save_dataset_iterative(dataset, output_dir, seed=42, label_column="labels"):
    """
    Splits the dataset into Train, Validation, and Test sets and saves them to disk.
    Applies iterative stratification on labels for label-balance in all splits. 
    
    Args:
        dataset (Dataset): The full processed dataset.
        output_dir (str): Directory path to save the dataset.
        seed (int): Random seed for reproducibility.
        label_column (str): name of column containing labels
        
    Returns:
        final_dataset (Dataset): HuggingFace dataset, splitted and stratified
    """

    # multi-hot-matrix for iterative stratification
    y_matrix = get_entity_presence_matric(dataset, label_column=label_column)
    
    # sklearn api expects (X, y) format
    # -> dummy X-array
    X = np.zeros(len(dataset))
    
    # split 1 - test set (10%)
    msss_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    # msss.split returns indices
    # -> only need first split
    train_val_idx, test_idx = next(msss_test.split(X, y_matrix))
    
    test_dataset = dataset.select(test_idx)
    remaining_dataset = dataset.select(train_val_idx)
    
    # slice y_matrix to match remaining dataset
    # for second split
    y_remaining = y_matrix[train_val_idx]
    X_remaining = X[train_val_idx]
    
    # split 2 - validation set (10%, approx. 1/9 of remaining dataset)
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.111, random_state=seed)
    train_idx, val_idx = next(msss_val.split(X_remaining, y_remaining))
    
    train_dataset = remaining_dataset.select(train_idx)
    val_dataset = remaining_dataset.select(val_idx)
    
    # combine into DatasetDict
    final_dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    # save to disk
    print(f"Saving iteratively stratified dataset to {output_dir}...")
    print(f"\tTrain: {len(train_dataset)} samples")
    print(f"\tValidation: {len(val_dataset)} samples")
    print(f"\tTest: {len(test_dataset)} samples")
    final_dataset.save_to_disk(output_dir)
    print("Save complete.")   
    
    return final_dataset


# SIMPLE STRATIFICATION BY CRITERIA TYPE
def split_and_save_dataset(dataset, output_dir, seed=42):
    """
    Splits the dataset into Train, Validation, and Test sets and saves them to disk.
    
    Strategy:
    - Test: 10% of data (Stratified)
    - Validation: 10% of data (Stratified)
    - Train: 80% of data (Stratified)

    Args:
        dataset (Dataset): The full processed dataset.
        output_dir (str): Directory path to save the dataset.
        seed (int): Random seed for reproducibility.

    Returns:
        None
    """
    
    # the dataset-library currently does not directly support a three-way split
    # therefore, splitting has to be done in two steps
    
    
    # first split - test set
    # stratify by criteria_type (balance inc/excl samples)
    if "criteria_type" in dataset.features:
        split_1 = dataset.train_test_split(test_size=0.1, stratify_by_column="criteria_type", seed=seed)
    else:
        split_1 = dataset.train_test_split(test_size=0.1, seed=seed)
    
    test_dataset = split_1["test"]
    remaining_dataset = split_1["train"]
    
    # second split - validation + train set
    # val should be 10% of total -> approx. 1/9th of remaining 90%
    if "criteria_type" in dataset.features:
        split_2 = remaining_dataset.train_test_split(test_size=0.111, stratify_by_column="criteria_type", seed=seed)
    else:
        split_2 = remaining_dataset.train_test_split(test_size=0.111, seed=seed)
        
    train_dataset = split_2["train"]
    val_dataset = split_2["test"]
    
    
    # splitted dataset
    final_dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    # save dataset to disk
    print(f"Saving split dataset to {output_dir}...")
    print(f"\tTrain: {len(train_dataset)} samples")
    print(f"\tValidation: {len(val_dataset)} samples")
    print(f"\tTest: {len(test_dataset)} samples")
    final_dataset.save_to_disk(output_dir)
    print("Save complete.")
    
    return None



    
    
# boilerplate
# %%
if __name__ == "__main__":
    
    DATA_DIR = "./data/chia_without_scope"
    OUTPUT_DIR = "./data/chia_without_scope_parsedNER_lines_v1_ITERATIVETEST"
    MODEL_CHECKPOINT = "emilyalsentzer/Bio_ClinicalBERT"
    
    # %%
    if not os.path.exists(DATA_DIR):
        print(f"Please place your .txt and .ann files in {DATA_DIR}")
    else:
        # load & process chia dataset
        dataset, label2id, id2label = load_chia_dataset(DATA_DIR, MODEL_CHECKPOINT)
        
        # simple stratification by criteria_type
        # # split & save processed dataset
        # if len(dataset) > 0:
        #     split_and_save_dataset(dataset, OUTPUT_DIR)
        #     save_label_map(label2id, id2label, OUTPUT_DIR)
        #     print("\n--- Pipeline Complete ---")
        #     print(f"Dataset saved to: {OUTPUT_DIR}")
        #     print(f"Number of distinct NER labels: {len(label2id)}")

        # iterative stratification by label
        # split & save processed dataset
        if len(dataset) > 0:
            split_and_save_dataset_iterative(dataset, OUTPUT_DIR)
            save_label_map(label2id, id2label, OUTPUT_DIR)
            print("\n--- Pipeline Complete ---")
            print(f"Dataset saved to: {OUTPUT_DIR}")
            print(f"Number of distinct NER labels: {len(label2id)}")
    
    # %%
    # example = dataset[150]
    # print(f"input ids: {example["input_ids"]}")
    # print(f"sentence text: {example["sentence_text"]}")
    # tokens = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT).convert_ids_to_tokens(example["input_ids"])
    # print(f"tokens: {tokens}")
    # label_ids = example["labels"]
    # label_names = list()
    # for id in label_ids:
    #     label_names.append(id2label[id])
    # print(f"labels: {label_names}")