# PARSING THE CHIA DATASET FOR NER TASKS

# %%
# imports
import os
import glob
import re
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


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