

# %%
# imports
import os
import json
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report



# %%
# functions
def load_label_map(input_dir, filename="label_map.json"):
    """
    Loads label mappings to ensure model config matches dataset.
    """
    
    path = os.path.join(input_dir, filename)
    with open(path, "r", encoding="UTF-8") as f:
        data = json.load(f)
    
    label2id = data["label2id"]
    id2label = data["id2label"]
    
    return label2id, id2label



def format_re_input(example):
    """
    Injects Entity Markers [E1], [/E1], [E2], [/E2] into the text.
    Must insert from right-to-left to prevent index shifting.
    """
    
    text = example["text"]
    
    # collect insertion indices
    insertions = [
        (example['e1_start'], "[E1]"), 
        (example['e1_end'],   "[/E1]"),
        (example['e2_start'], "[E2]"), 
        (example['e2_end'],   "[/E2]")
    ]
    
    # sort insertion points descending
    # --> for insertion right to left
    insertions.sort(key=lambda x: x[0], reverse=True)
    
    # entity marker injeciton
    marked_text = text
    for pos, marker in insertions:
        marked_text = marked_text[:pos] + marker + marked_text[pos:]
    
    return {"text_with_markers": marked_text}



def compute_metrics(eval_pred):
    """
    Computes precision, recall, f1, and accuracy.
    Uses 'weighted' average to account for class imbalance (NO_RELATION dominance).
    """
    
    logits, true_labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(true_labels, predictions)
    
    results = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
        }
    
    return results

# %% 
# configuration

# # local paths
# BASE_PATH = "./data"
# DATA_PATH = os.path.join(BASE_PATH, "chia_without_scope_parsedRE_test_small_fullDownsampled")
# OUTPUT_DIR = "./models/re_test_small_fullDownsampled2"
# MODEL_CHECKPOINT = "emilyalsentzer/Bio_ClinicalBERT"

# paths for google colab
BASE_PATH = "/content/drive/MyDrive/masters_thesis_computing"
DATA_PATH = os.path.join(BASE_PATH, "data/chia_without_scope_parsedRE_test")
print(f"DATA_PATH = {DATA_PATH}")
OUTPUT_DIR = os.path.join(BASE_PATH, "models/RE_chia_test_small_fullDownsampled")
MODEL_CHECKPOINT = "emilyalsentzer/Bio_ClinicalBERT"

# %%
# Hyperparameters
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5



def main():
    """
    Training loop.
    """
    
    # %%
    # load training data + label mapping
    print(f"Loading dataset from {DATA_PATH}...")
    try:
        dataset = load_from_disk(DATA_PATH)
        label2id, id2label = load_label_map(DATA_PATH)
    except FileNotFoundError:
        print("Error: Dataset or label_map.json not found.")
        # return

    print(f"Loaded {len(dataset["train"])} training samples.")
    print(f"Labels: {list(label2id.keys())}")
    
    
    # %%
    # entity marker injection
    print("Injecting entity markers...")
    dataset = dataset.map(format_re_input)
    
    
    # %%
    # tokenizer setup
    print("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # add speical tokens to tokenizer
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added_tokens} special tokens: {special_tokens["additional_special_tokens"]}")
    
    # tokenization
    def tokenize_function(examples):
        """Helper function for tokenization."""
        return tokenizer(
            examples["text_with_markers"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        )

    print("Tokenizing...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # %%
    # remove columns not needed for training
    columns_to_keep = ["input_ids", "attention_mask", "label"]
    tokenized_dataset.set_format(type="torch", columns=columns_to_keep)
    
    
    # %%
    # model initialization
    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    # resize embeddings to fit new special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # %%
    # training configuration
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )
    
    # %%
    # training
    print("Starting training...")
    trainer.train()
    
    # %%
    # evaluation on test set
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(tokenized_dataset["test"])
    print("\n--- evaluation results ---")
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")
    
    # detailed classificaiton report
    print("\n--- detailed classification report ---")
    raw_pred, _, _ = trainer.predict(tokenized_dataset["test"])
    y_pred = np.argmax(raw_pred, axis=1)
    y_true = tokenized_dataset["test"]["label"]
    
    print(classification_report(
        y_true,
        y_pred,
        target_names=list(label2id.keys()),
        zero_division=0
    ))
    
    # %%
    # save tokenizer trained model
    print(f"Saving trained model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    

# %%
# boilerplate
if __name__ == "__main__":

    main()