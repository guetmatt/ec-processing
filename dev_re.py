# DEVELOPMENT OF RELATION EXTRACTION MODULE
# rough plan:
# BERT-based re with chia-dataset
# then gen-ai on top for logical composition/json generation or similar

# generated with the help of ChatGPT


# %%
# imports
import json
import itertools
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# %%
# utility - entity markers
def insert_entity_markers(sentence, e1, e2):
    entities = sorted([e1, e2], key=lambda x: x["start"], reverse=True)
    text = sentence
    
    for idx, ent in enumerate(entities):
        open_tag = f"[E{idx+1}]"
        close_tag = f"[/E{idx+1}]"
        text = (
            text[:ent["start"]] +
            open_tag +
            text[ent["start"]:ent["end"]] +
            close_tag + 
            text[ent["end"]:]
        )
        
    return text

