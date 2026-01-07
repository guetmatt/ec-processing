# TO DO
# - parse complete dataset
# - look at labels used in chia annotation -- decide if some should be merged or removed
# - develop NER training/evaluation pipeline

# Current format of processed eligibility criterion:
# ['[CLS]', 'patients', 'with', 's', '##ym', '##pt', '##oma', '##tic', 'c', '##ns', 'meta', '##sta', '##ses', 'or', 'le', '##pt', '##ome', '##ning', '##eal', 'involvement', 'patients', 'with', 'known', 'brain', 'meta', '##sta', '##ses', ',', 'unless', 'these', 'meta', '##sta', '##ses', 'have', 'been', 'treated', 'and', '/', 'or', 'have', 'been', 'stable', 'for', 'at', 'least', 'six', 'months', 'prior', 'to', 'study', 'start', '.', 'subjects', 'with', 'a', 'history', 'of', 'brain', 'meta', '##sta', '##ses', 'must', 'have', 'a', 'head', 'c', '##t', 'with', 'contrast', 'to', 'document', 'either', 'response', 'or', 'progression', '.', 'patients', 'with', 'bone', 'meta', '##sta', '##ses', 'as', 'the', 'only', 'site', '(', 's', ')', 'of', 'me', '##as', '##urable', 'disease', 'patients', 'with', 'he', '##pa', '##tic', 'artery', 'ch', '##em', '##oe', '##mbo', '##li', '##zation', 'within', 'the', 'last', '6', 'months', '(', 'one', 'month', 'if', 'there', 'are', 'other', 'sites', 'of', 'me', '##as', '##urable', 'disease', ')', 'patients', 'who', 'have', 'been', 'previously', 'treated', 'with', 'radioactive', 'directed', 'the', '##rap', '##ies', 'patients', 'who', 'have', 'been', 'previously', 'treated', 'with', 'e', '##pot', '##hil', '##one', 'patients', 'with', 'any', 'peripheral', 'ne', '##uro', '##pathy', 'or', 'un', '##res', '##ol', '##ved', 'di', '##ar', '##r', '##hea', 'greater', 'than', 'grade', '1', 'patients', 'with', 'severe', 'cardiac', 'ins', '##uff', '##iciency', 'patients', 'taking', 'co', '##uma', '##din', 'or', 'other', 'war', '##fari', '##n', '-', 'containing', 'agents', 'with', 'the', 'exception', 'of', 'low', 'dose', 'war', '##fari', '##n', '(', '1', 'mg', 'or', 'less', ')', 'for', 'the', 'maintenance', 'of', 'in', '-', 'dwelling', 'lines', 'or', 'ports', 'patients', 'taking', 'any', 'experimental', 'the', '##rap', '##ies', 'history', 'of', 'another', 'ma', '##li', '##gna', '##ncy', 'within', '5', 'years', 'prior', 'to', 'study', 'entry', 'except', 'cu', '##rative', '##ly', 'treated', 'non', '-', 'me', '##lan', '##oma', 'skin', 'cancer', ',', 'pro', '##state', 'cancer', ',', 'or', 'c', '##er', '##vic', '##al', 'cancer', 'in', 'sit', '##u', 'patients', 'with', 'active', 'or', 'suspected', 'acute', 'or', 'chronic', 'un', '##con', '##tro', '##lled', 'infection', 'including', 'a', '##b', '##cess', '##es', 'or', 'fist', '##ula', '##e', 'patients', 'with', 'a', 'medical', 'or', 'psychiatric', 'illness', 'that', 'would', 'pre', '##c', '##lude', 'study', 'or', 'informed', 'consent', 'and', '/', 'or', 'history', 'of', 'non', '##com', '##p', '##liance', 'to', 'medical', 'regime', '##ns', 'or', 'inability', 'or', 'unwilling', '##ness', 'to', 'return', 'for', 'all', 'scheduled', 'visits', 'hi', '##v', '+', 'patients', 'pregnant', 'or', 'la', '##ct', '##ating', 'females', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
# - do I want to process sentence-wise or field-wise?



# PARSING THE CHIA DATASET FOR NER TASKS - CRITERIA-WISE APPROACH

# %%
# imports
import os
import json
import glob
import re
import pandas as pd
from datasets import Dataset, ClassLabel, DatasetDict
from transformers import AutoTokenizer


# %%
# function definitions
def parse_brat_file(ann_path):
    """
    Parses a single .ann file and extracts entity annotations.

    Args:
        ann_path (str): Path to the .ann file.
    
    Returns:
        data (list): A list of dictionaries, each representing an entity with
                "id", "type", "start", "end", and "text".
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
            
            

def align_tokens_and_labels(text, entities, tokenizer, label2id):
    """
    Tokenizes text and maps character-level entities to token-level BIO labels.
    Updates label2id in-place when new labels are found.

    Args:
        text (str): clinical eligibility criteria text
        entities (list): list of entity dictionaries
        tokenizer: instance of AutoTokenizer
        label2id (dict): dictionary mapping label names to IDs 
    
    Returns:
        data_tokenized_aligned (dict): A dictionary containing input_ids, attention_mask, and aligned labels
    """
    
    tokenized_inputs = tokenizer(
        text, 
        truncation=True, 
        max_length=512, 
        return_offsets_mapping=True, 
        padding="max_length"
    )
    
    offset_mapping = tokenized_inputs["offset_mapping"]
    # Initialize all labels to 'O' (Outside)
    labels = [label2id["O"]] * len(tokenized_inputs["input_ids"])
    
    for entity in entities:
        entity_type = entity["type"]
        start_char = entity["start"]
        end_char = entity["end"]
        
        # define BIO labels
        b_label = f"B-{entity_type}"
        i_label = f"I-{entity_type}"
        
        # update label map when new label is found
        if b_label not in label2id:
            label2id[b_label] = len(label2id)
        if i_label not in label2id:
            label2id[i_label] = len(label2id)
            
        b_id = label2id[b_label]
        i_id = label2id[i_label]
        
        found_start = False
        
        # iterate through tokens to find those covering the entity span
        for idx, (offset_start, offset_end) in enumerate(offset_mapping):
            # skip special tokens
            if offset_start == 0 and offset_end == 0: continue 
            
            # check for overlap
            if offset_start >= start_char and offset_end <= end_char:
                if not found_start:
                    # update label to B-<TYPE> for first entity token
                    labels[idx] = b_id
                    found_start = True
                else:
                    # update label to I-<TYPE> for subsequent entity tokens  
                    labels[idx] = i_id
            
            # edge case where token boundary not perfect
            # (e.g. due to subword generation by tokenizer)
            # treat as beginning of entity
            elif offset_start < start_char and offset_end > start_char:
                # update label to B-<TYPE> for first entity token
                labels[idx] = b_id
                found_start = True
    
    
    # generate dictionary of tokenized text with aligned labels
    data_tokenized_aligned = {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": labels
    } 
    
    return data_tokenized_aligned 



def load_chia_dataset(data_dir, model_checkpoint):
    """
    Main driver function: Loads and processes files of chia dataset.
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
    data_samples = list()
    
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
        
        # retrieve criteria type
        # default to "unknown"
        file_name = os.path.basename(txt_path)
        criteria_type = "unknown"
        if "_exc" in file_name:
            criteria_type = "exclusion"
        elif "_inc" in file_name:
            criteria_type = "inclusion"
            
        # read eligibility criteria content
        with open(txt_path, "r", encoding="UTF-8") as f:
            text = f.read()
        
        # parse named entities
        entities = parse_brat_file(ann_path)
        
        # align and tokenize text and ner-labels
        # label2id is updated in-place
        processed_sample = align_tokens_and_labels(text, entities, tokenizer, label2id)

        # get file name for reference
        processed_sample["file_name"] = file_name
        processed_sample["criteria_type"] = criteria_type
        data_samples.append(processed_sample)
        
    # convert list of processed files to HuggingFace dataset
    # via pandas for easier conversion    
    df = pd.DataFrame(data_samples)
    hf_dataset = Dataset.from_pandas(df)
    
    # convert criteria_type to ClassLabel
    # 0 -> exclusion, 1 -> inclusion, 2 -> unknown
    features = hf_dataset.features.copy()
    features["criteria_type"] = ClassLabel(names=["exclusion", "inclusion", "unknown"])
    hf_dataset = hf_dataset.cast_column("criteria_type", features["criteria_type"])
    
    # reverse label mapping for decoding label2id
    id2label = {v: k for k, v in label2id.items()}
    
    return hf_dataset, label2id, id2label



def save_label_map(label2id, id2label, output_dir, filename="label_map.json"):
    """
    Saves label mappings to a JSON file.
    
    Args:
        label2id (dict): Mapping from label name (str) to id (int)
        id2label (dict): Mapping from id (int) to label name (str)
        output_dir (str): Directory where the file should be saved
        filename (str): Name of the output file
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



# def load_label_map(input_dir, filename="label_map.json"):
#     """
#     Helper function to load label mappings.

#     Args:
#         input_dir (str): path of directory with label mapping file
#         filename (str): name of the label mapping file
#     """
    
#     path = os.path.join(input_dir, filename)
    
#     with open(path, "r", encoding="UTF-8") as f:
#         data = json.load(f)
    
#     label2id = data["label2id"]
#     # convert keys to int (json saves keys as strings)
#     id2label = {int(k): v for k, v in data["id2label"].items()}
    
#     return label2id, id2label
    
    
        
def save_dataset_to_disk(dataset, output_dir):
    """
    Saves a HuggingFace dataset to a specified directory.
    
    Args:
        dataset (HF dataset): a HuggingFace dataset object
        output_dir (str): the directory path where the dataset data is saved to
    """
    
    print(f"Saving dataset to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"Dataset saved to {output_dir}")
    
    return None    



# stratify by label instead of criteria type????
def split_and_save_dataset(dataset, output_dir, seed=42):
    """
    Splits the dataset into train (80%), validation (10%), and test (10%).
    Saves splitted data to disk as a DatasetDict.
    
    ...
    """
    
    # the dataset-library currently does not directly support a three-way split
    # thereofe, splitting has to be done in two steps
    
    # first split - test set
    # stratify by criteria_type (balance inc/excl samples)
    if "criteria_type" in dataset.features:
        print("Splitting dataset with stratification by 'criteria_type'...")
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
    
    final_dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    # save splitted dataset to disk
    print(f"Saving splitted dataset to {output_dir}...")
    print(f"\tTrain: {len(train_dataset)} samples")
    print(f"\tValidation: {len(val_dataset)} samples")
    print(f"\tTest: {len(test_dataset)} samples")
    
    final_dataset.save_to_disk(output_dir)
    print(f"Save completed.")
    
    return None
    
    


# def load_dataset_from_disk(input_dir):
#     """
#     Loads a saved HuggingFace dataset from disk.
    
#     Args:
#         input_dir (str): the directory path where the dataset is saved
        
#     Returns:
#         dataset (HuggingFace dataset): loaded dataset object
#     """
    
#     try:
#         dataset = load_from_disk(input_dir)
#         print(f"Loaded dataset from '{input_dir}' with {len(dataset)} samples.")
#         return dataset
#     except FileNotFoundError:
#         print(f"Error: No dataset found at '{input_dir}'")
#         return None

    
    


# boilerplate
# %%
if __name__ == "__main__":
    
    # directory containing chia .txt and .ann files
    DATA_DIR = "./data/chia_without_scope_test"
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Note: Created empty directory {DATA_DIR} for data. Please add chia .txt and .ann files to proceed.")
    
    # %%
    # model
    model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
    
    # %%
    # run loading pipeline
    dataset, label2id, id2label = load_chia_dataset(DATA_DIR, model_checkpoint)
    
    
    # %%
    if len(dataset) > 0:
        print("\n--- Pipeline completed successfully! ---")
        print(f"Loaded {len(dataset)} samples.")
        print(f"Labels found: {len(label2id)}")
        print(f"Label map:  {label2id}")
        
        print("\n Dataset:")
        print(dataset)
    
        
        # example: verifying specific labels from data
        if "B-Condition" in label2id:
            print(f"Verification: ID for B-Condition: {label2id['B-Condition']}")
            
            

    # %%
    # save dataset to disk
    OUTPUT_DIR = "./data/chia_without_scope_parsedNER_v1"
    if len (dataset) > 0:
        split_and_save_dataset(dataset, OUTPUT_DIR)
        save_label_map(label2id, id2label, OUTPUT_DIR)
    
    
    
    # save_dataset_to_disk(dataset, OUTPUT_DIR)
    # save_label_map(label2id, id2label, OUTPUT_DIR)
    
    # %%
    # load dataset test
    # data = load_dataset_from_disk(OUTPUT_DIR)
    # label2id_loaded, id2label_loaded = load_label_map(OUTPUT_DIR)
    # if data:
    #     print(f"\nVerification Successful:")
    #     print(f"Samples: {len(data)}")
    #     # Check features to ensure metadata (criteria_type) was preserved
    #     print(f"Features: {data.features}")
    
    
    # %%
    example = dataset[0]
    input_ids = example["input_ids"]
    tokens = AutoTokenizer.from_pretrained(model_checkpoint).convert_ids_to_tokens(input_ids)
    print(tokens)

    # %%
    label_ids = dataset[0]["labels"]
    label_names = list()
    for id in label_ids:
        label_names.append(id2label[id])
    
    # %%
    zipped = list(zip(tokens, label_names))
    
    for el in zipped:
        print(el)