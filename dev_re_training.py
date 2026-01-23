

# %%
# imports
import os
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from functools import partial
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


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



def compute_weighted_loss(outputs, labels, num_items_in_batch=None, class_weights=None):
    """
    Custom weighted loss function to handle class imbalance.
    Dynamically moves weights to same device as input.
    """    

    # extract logits
    logits = outputs.get("logits")
    
    # move class_weights to correct device
    if class_weights is not None:
        class_weights = class_weights.to(logits.device)
    
    # define and compute weighted loss
    loss_func = CrossEntropyLoss(weight=class_weights)
    loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
    
    return loss



def plot_confusion_matrix(y_true, y_pred, label_map, output_dir):
    """Generates and saves a normalized confusion matrix heatmap."""

    # sort labels by ID --> axis alignment matches the matrix indices
    sorted_labels = sorted(label_map.items(), key=lambda x: x[1])
    label_names = [x[0] for x in sorted_labels]
    label_ids = [x[1] for x in sorted_labels]

    # compute matrix
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids, normalize="true")
    
    # plot matrix
    plt.figure(figsize=(14, 22))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="Blues",
        cbar=True
    )
    
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.title("Normalized Confusion Matrix (Relation Extraction)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # save plot to disk
    save_path = os.path.join(output_dir, "confusion_matrix_re.png")
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()
    
    return None


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
OUTPUT_DIR = os.path.join(BASE_PATH, "models/RE_chia_test")
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
    
    
    
    # %%
    # weighted loss  
    # calculate class weights
    # based on class imbalance
    train_labels = dataset["train"]["label"]
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    # dampen class_weights 
    # to reduce risk of gradient explosion
    class_weights_np = np.sqrt(class_weights_np)
    # re-normalize class_weigts
    # so that average weight is roguhly 1.0
    class_weights_np = class_weights_np / np.mean(class_weights_np)
    class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32).to(model.device)
    
    print("\n --- WEIGHTING SANITY CHECK ---")
    for idx, weight in enumerate(class_weights_np):
        label_name = id2label[str(idx)]
        print(f"Class '{label_name}': Weight {weight:.4f}")
    max_w = np.max(class_weights_np)
    min_w = np.min(class_weights_np)
    ratio = max_w / min_w
    
    print(f"\nMax Weight: {max_w:.2f}")
    print(f"Min Weight: {min_w:.2f}")
    print(f"Penalty Ratio: {ratio:.2f}x")
    print(f"(This means the model is penalized {ratio:.0f} times more for missing a rare class than a common one)")
    
    if ratio > 50:
        print("\n[WARNING] Your penalty ratio is very high (>50x). Risk of gradient explosion.")
        print("Consider 'Gradient Clipping' or using sqrt(weights) to soften them.")
    print("-------------------------------------\n")
    
    
    # custom weighted loss function
    # partial --> freezes arguments
    # --> so they do not have to be passed to Trainer
    weighted_loss = partial(compute_weighted_loss, class_weights=class_weights_tensor)
  
    # %%
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        compute_loss_func=weighted_loss
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
    
    # ensures that all labels are used for evaluation report
    # even if they do not occur in data 
    all_labels_ids = list(label2id.values())
    all_labels_names = list(label2id.keys())
    
    print(classification_report(
        y_true,
        y_pred,
        labels=all_labels_ids,
        target_names=all_labels_names,
        zero_division=0
    ))
    
    
    # generate confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, label2id, OUTPUT_DIR)
    
    
    # %%
    # save tokenizer trained model
    print(f"Saving trained model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    

# %%
# boilerplate
if __name__ == "__main__":

    main()