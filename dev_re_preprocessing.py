# DEVELOPMENT OF RELATION EXTRACTION MODULE
# rough plan:
# BERT-based re with chia-dataset
# MAYBE then gen-ai on top for logical composition/json generation or similar

# (1) LABEL ENCODING
# (2) TOKENIZATIONM


# %%
# imports
import json
import os
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer


# %%
# configuration
INPUT_DIR = "./data/chia_parsed_v1"
OUTPUT_DIR = "./data/chia_tokenized_v1"
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 256


# split data into train/test splits
def split_dataset(dataset, test_size=0.2, seed=42):
    """
    Splits the loaded dataset into Train and Test only.
    """
    
    try:
        split = dataset.train_test_split(
            test_size=test_size, 
            seed=seed,
            stratify_by_column="label" 
        )
    except ValueError:
        print("Warning: Stratification failed (Classes likely to small). Splitting randomly.")
        split = dataset.train_test_split(
            test_size=test_size,
            seed=seed)

    ds_splitted = DatasetDict({
        "train": split["train"],
        "test": split["test"]
    })
    
    return ds_splitted

# %%
# label mapping, str <-> int
def create_label_map(dataset):
    """
    Scans the training set to create a label-to-id mapping.
    Ensures 'NO_RELATION' is always index 0.
    """
    # extract set of labels
    unique_labels = set(dataset["train"]["label"])
    
    # sort for deterministic ordering
    sorted_labels = sorted(list(unique_labels))
    
    # force label "NO_RELATION" = 0
    # MENTION IN PAPER!
    if "NO_RELATION" in sorted_labels:
        sorted_labels.remove("NO_RELATION")
        sorted_labels.insert(0, "NO_RELATION")
        
    # create mapping dicts
    label2id = {label: i for i, label in enumerate(sorted_labels)}
    id2label = {i: label for i, label in enumerate(sorted_labels)}
    
    return label2id, id2label


# %%
# tokenization
def tokenize_function(examples, tokenizer, label2id):
    """
    Tokenizes text and maps labels to integers.
    """
    # tokenize the 'marked_text' column
    tokenized_inputs = tokenizer(
        examples["marked_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )
    
    # map string labels to integers
    tokenized_inputs["labels"] = [label2id[label] for label in examples["label"]]
    
    return tokenized_inputs


# %%
# complete process of preprocessing, tokenization and saving
def main():
    print(f"Loading dataset from '{INPUT_DIR}'...")
    try:
        dataset = load_from_disk(INPUT_DIR)
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {INPUT_DIR}. Run the split script first.")
        return
    
    # split dataset into train/test split
    print("Splitting dataset...")
    dataset = split_dataset(dataset, test_size=0.2, seed=42)
    print(f" Train: {len(dataset["train"])} | Test: {len(dataset["test"])}")
    
    # label mapping
    # save mapping to disk for later use
    print("Generating label map...")
    label2id, id2label = create_label_map(dataset)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=4)
    print(f"Label map saved (Found {len(label2id)} classes).")

    # tokenizer setup
    print(f"Initializing tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # add special tokens for entity markers
    new_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"Added {num_added_toks} special tokens: {new_tokens}")

    # tokenization
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer, label2id),
        batched=True,
        # remove the original text columns
        # saves space & prevents PyTorch collation errors
        remove_columns=["text", "marked_text", "e1", "e2", "label", "criterion_id"]
    )

    # save preprocessed dataset and tokenizer to disk
    # save tokenizer to load added tokens later
    print(f"Saving tokenized data to '{OUTPUT_DIR}'...")
    tokenized_datasets.save_to_disk(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR) 
    
    print("\nPROCESS COMPLETE.")
    print(f"\tTrain shape: {tokenized_datasets["train"].shape}")
    print(f"\tTest shape:  {tokenized_datasets["test"].shape}")


# %%
dataset = load_from_disk(INPUT_DIR)

# %%
# DATASET muss noch in train/test(/val) gesplittet werden



# %%
if __name__ == "__main__":
    main()