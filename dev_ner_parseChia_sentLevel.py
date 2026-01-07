# TO DO
# - parse complete dataset
# - look at labels used in chia annotation -- decide if some should be merged or removed
# - develop NER training/evaluation pipeline

# Current format of processed eligibility criterion:
# ['[CLS]', 'patients', 'with', 's', '##ym', '##pt', '##oma', '##tic', 'c', '##ns', 'meta', '##sta', '##ses', 'or', 'le', '##pt', '##ome', '##ning', '##eal', 'involvement', 'patients', 'with', 'known', 'brain', 'meta', '##sta', '##ses', ',', 'unless', 'these', 'meta', '##sta', '##ses', 'have', 'been', 'treated', 'and', '/', 'or', 'have', 'been', 'stable', 'for', 'at', 'least', 'six', 'months', 'prior', 'to', 'study', 'start', '.', 'subjects', 'with', 'a', 'history', 'of', 'brain', 'meta', '##sta', '##ses', 'must', 'have', 'a', 'head', 'c', '##t', 'with', 'contrast', 'to', 'document', 'either', 'response', 'or', 'progression', '.', 'patients', 'with', 'bone', 'meta', '##sta', '##ses', 'as', 'the', 'only', 'site', '(', 's', ')', 'of', 'me', '##as', '##urable', 'disease', 'patients', 'with', 'he', '##pa', '##tic', 'artery', 'ch', '##em', '##oe', '##mbo', '##li', '##zation', 'within', 'the', 'last', '6', 'months', '(', 'one', 'month', 'if', 'there', 'are', 'other', 'sites', 'of', 'me', '##as', '##urable', 'disease', ')', 'patients', 'who', 'have', 'been', 'previously', 'treated', 'with', 'radioactive', 'directed', 'the', '##rap', '##ies', 'patients', 'who', 'have', 'been', 'previously', 'treated', 'with', 'e', '##pot', '##hil', '##one', 'patients', 'with', 'any', 'peripheral', 'ne', '##uro', '##pathy', 'or', 'un', '##res', '##ol', '##ved', 'di', '##ar', '##r', '##hea', 'greater', 'than', 'grade', '1', 'patients', 'with', 'severe', 'cardiac', 'ins', '##uff', '##iciency', 'patients', 'taking', 'co', '##uma', '##din', 'or', 'other', 'war', '##fari', '##n', '-', 'containing', 'agents', 'with', 'the', 'exception', 'of', 'low', 'dose', 'war', '##fari', '##n', '(', '1', 'mg', 'or', 'less', ')', 'for', 'the', 'maintenance', 'of', 'in', '-', 'dwelling', 'lines', 'or', 'ports', 'patients', 'taking', 'any', 'experimental', 'the', '##rap', '##ies', 'history', 'of', 'another', 'ma', '##li', '##gna', '##ncy', 'within', '5', 'years', 'prior', 'to', 'study', 'entry', 'except', 'cu', '##rative', '##ly', 'treated', 'non', '-', 'me', '##lan', '##oma', 'skin', 'cancer', ',', 'pro', '##state', 'cancer', ',', 'or', 'c', '##er', '##vic', '##al', 'cancer', 'in', 'sit', '##u', 'patients', 'with', 'active', 'or', 'suspected', 'acute', 'or', 'chronic', 'un', '##con', '##tro', '##lled', 'infection', 'including', 'a', '##b', '##cess', '##es', 'or', 'fist', '##ula', '##e', 'patients', 'with', 'a', 'medical', 'or', 'psychiatric', 'illness', 'that', 'would', 'pre', '##c', '##lude', 'study', 'or', 'informed', 'consent', 'and', '/', 'or', 'history', 'of', 'non', '##com', '##p', '##liance', 'to', 'medical', 'regime', '##ns', 'or', 'inability', 'or', 'unwilling', '##ness', 'to', 'return', 'for', 'all', 'scheduled', 'visits', 'hi', '##v', '+', 'patients', 'pregnant', 'or', 'la', '##ct', '##ating', 'females', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
# - do I want to process sentence-wise or field-wise?



# PARSING THE CHIA DATASET FOR NER TASKS - SENTENCE-WISE APPROACH

# %%
# imports
import os
import glob
import re
import json
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer


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



# stratify by label instead of criteria type????
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
    
    DATA_DIR = "./data/chia_without_scope_test"
    OUTPUT_DIR = "./data/chia_without_scope_parsedNER_lines_v1"
    MODEL_CHECKPOINT = "emilyalsentzer/Bio_ClinicalBERT"
    
    # %%
    if not os.path.exists(DATA_DIR):
        print(f"Please place your .txt and .ann files in {DATA_DIR}")
    else:
        # load & process chia dataset
        dataset, label2id, id2label = load_chia_dataset(DATA_DIR, MODEL_CHECKPOINT)
        
        # split & save processed dataset
        if len(dataset) > 0:
            split_and_save_dataset(dataset, OUTPUT_DIR)
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