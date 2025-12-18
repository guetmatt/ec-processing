# based on: https://huggingface.co/learn/llm-course/chapter7/2?fw=pt
# and: https://huggingface.co/docs/transformers/main/tasks/token_classification

# imports
from transformers import AutoTokenizer
from datasets import Dataset
from functools import partial




def read_data(filepath: str):
    with open(filepath, "r", encoding="UTF-8") as f:
        lines = f.readlines()
    return lines


def structure_clinicalTrialParserDataset(data_lines: list):
    """
    Structure data that has been read into lines with continuous strings
    into a dictionary with named elements.
    This function is specifically designed for preprocessing
    the dataset "Clinical Trial Parser" by Tseo et al. (2020).
    Example entry data_dict:
    {
        'text': 'has a significant risk for suicide',
        'entities': 
            [{
                'start': 27,
                'end': 35,
                'label': 'chronic_disease',
                'entity_text': 'suicide'
            }]}
    Args:
        data_lines (list): _description_

    Returns:
        data_dict (dict): _description_
    """
    data_dict = dict()
    
    for idx, line in enumerate(data_lines):
        line_split = line.split("\t")
        text = line_split[1]
        # split(",") only applies when multiple entities annotated
        # annotation = "number:number:label"
        annotation = line_split[0].split(",")
        
        data_dict[idx] = {
            "text": text,
            "entities": list()
        }
        
        for anno in annotation:
            annotation_split = anno.split(":")
            
            if len(annotation_split) %3 != 0:
                print(f"Error in line {idx}: {line_split[0]}.\n Expected list with three elements.")
                break  
            
            data_dict[idx]["entities"].append(
                {
                    "span_start": int(annotation_split[0])-1, # -1, original data indexing starts at 1
                    "span_end": int(annotation_split[1]),
                    "label": annotation_split[2],
                    "entity_text": text[int(annotation_split[0])-1:int(annotation_split[1])]
                }
            )
    
    return data_dict


def bioner_clinicalTrialParserDataset(data_dict: dict):
    """
    Restructure the structured Clinical Trial Parser Dataset
    to the NER-BIO labeling scheme.
    This function is specifically designed to convert 
    the output from function "structure_clinicalTrialParserDataset"
    to the NER-BIO scheme.

    Args:
        data_dict (dict): _description_

    Returns:
        data_bioscheme (dict): {
            "text": ['has', 'a', 'significant', 'risk', 'for', 'suicide'],
            "labels": ['O', 'O', 'O', 'O', 'O', 'B-chronic_disease']
        }
    """
    
    data_bioscheme = dict()
    for key, val in data_dict.items():
        text = val["text"]
        entities = val["entities"]
        
        tokens = list()
        token_spans = list()
        
        # get token and token indices
        pos = 0
        for tok in text.split():
            tok_start = text.index(tok, pos)
            tok_end = tok_start + len(tok)
            tokens.append(tok)
            token_spans.append((tok_start, tok_end))
            pos = tok_end
        
        # initiate list of NER-labels corresponding to tokens
        # "O" will be replaced in list if entity
        labels = ["O"] * len(tokens)
        
        # assign NER labels in BIO scheme
        for ent in entities:
            ent_start = ent["span_start"]
            ent_end = ent["span_end"]
            ent_label = ent["label"]
            
            # if token span is in entity span from original dataset
            # --> NER label
            first = True
            for idx, (tok_start, tok_end) in enumerate(token_spans):
                if tok_start >= ent_start and tok_end <= ent_end:
                    if first:
                        labels[idx] = f"B-{ent_label}"
                        first = False
                    else:
                        labels[idx] = f"I-{ent_label}"
        
        data_bioscheme[key] = {
            "text": tokens,
            "labels": labels
        }

    return data_bioscheme



def create_dataset(data_dict: dict):
    """
    Converts a dataset dictionary to the Dataset class
    from the package "datasets" used in applications
    of the "Huggingface Transformers" library. 

    Args:
        data_dict (dict): _description_

    Returns:
        dataset (Dataset): _description_
    """
    
    text_list = list()
    labels_list = list()
    
    for key, val in data_dict.items():
        text_list.append((val["text"]))
        labels_list.append((val["labels"]))
    data_restructured = {
        "text": text_list,
        "labels": labels_list
    }
    
    dataset = Dataset.from_dict(data_restructured)
    
    return dataset


def align_tokens_with_labels(dataset: Dataset, tokenizer_model="emilyalsentzer/Bio_ClinicalBERT"):
    """
    Tokens and labels have to be realigned after tokenization
    into sub-tokens. Only the first subtoken of a token is
    labeled with an NER-label, following subtokens of the same token
    are labeld with "-100". This is common practice. 
    Returns a tokenized dataset with aligned labels.

    Args:
        dataset (datasets.Dataset): _description_

    Returns:
        tokenized_dataset: _description_
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    dataset_tokenized = tokenizer(dataset["text"], truncation=True, is_split_into_words=True)
    
    label_names_list = list()
    for idx, label in enumerate(dataset["labels"]):
        word_ids = dataset_tokenized.word_ids(batch_index=idx)  # Map tokens to their respective word.
        previous_word_idx = None
        label_names = list()
        
        for word_idx in word_ids:
            if word_idx is None:
                label_names.append("-100")
            elif word_idx != previous_word_idx:
                label_names.append(label[word_idx])
            else:
                label_names.append("-100")
            previous_word_idx = word_idx
        
        label_names_list.append(label_names)        
        
    dataset_tokenized["label_names"] = label_names_list

    return dataset_tokenized




def prepare_datastructures_for_training(dataset: Dataset, tokenizer_model="emilyalsentzer/Bio_ClinicalBERT"):
    
    align_function = partial(align_tokens_with_labels, tokenizer_model=tokenizer_model)
    ds_tokenized = dataset.map(align_function, batched=True)

    ner_labels = set()
    for entry in ds_tokenized:
        ner_labels = ner_labels.union(set(entry["label_names"]))
            
    # reverse -> "O" in front = 0
    ner_labels = sorted(list(ner_labels), reverse=True)
    ner_labels.remove("-100")
    id2label = dict()
    label2id = dict()
    for index, label in enumerate(ner_labels):
        id2label[index] = label
        label2id[label] = index
    id2label[-100] = "-100"
    label2id["-100"] = -100
    
    # add new column to dataset
    # column "labels" containing label_ids corresponding to label_names
    # transformers expects "labels" column to contain int
    labels = list()
    for entry in ds_tokenized:
        label_ids = list()
        for label in entry["label_names"]:
            label_ids.append(label2id[label])
        labels.append(label_ids)
    ds_tokenized = ds_tokenized.remove_columns("labels")
    ds_tokenized = ds_tokenized.add_column("labels", labels)
    # remove columns "text" and "label_names"
    # to adhere to expected dataset format
    ds_tokenized = ds_tokenized.remove_columns("text")
    ds_tokenized = ds_tokenized.remove_columns("label_names")
    
    return ds_tokenized, id2label, label2id




def preprocess_ctp(filepath: str, tokenizer_model="emilyalsentzer/Bio_ClinicalBERT"):
    
    data_lines = read_data(filepath)
    data_dict = structure_clinicalTrialParserDataset(data_lines)
    data = bioner_clinicalTrialParserDataset(data_dict)
    dataset = create_dataset(data)    
    ds_tokenized, id2label, label2id = prepare_datastructures_for_training(dataset, tokenizer_model=tokenizer_model)
    
    return ds_tokenized, id2label, label2id









# boilerplate
if __name__ == "__main__":
    pass

    # ds_tok, id2label, label2id = preprocess_ctp("test", tokenizer_model="emilyalsentzer/Bio_ClinicalBERT")