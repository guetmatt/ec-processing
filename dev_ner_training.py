# %%
# imports
import os
import json
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report



# %%
# functions

def load_label_map(input_dir, filename="label_map.json"):
    """
    Helper function to load label mappings.

    Args:
        input_dir (str): path of directory with label mapping file
        filename (str): name of the label mapping file
    """
    
    path = os.path.join(input_dir, filename)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label mapping file not found at {path}")
    
    with open(path, "r", encoding="UTF-8") as f:
        data = json.load(f)
    
    label2id = data["label2id"]
    # json saves keys as strings -> convert to int
    id2label = {int(k): v for k, v in data["id2label"].items()}
    
    return label2id, id2label



def compute_metrics_seq(p):
    """
    Computes precision, recall, F1 using the seqeval library.
    Sequence-based evaluation, only counts predictions that
    match the entire entity span as correct predictions.
    
    Args:
        p: predictions and labels
    Returns:
        results_overall (dict): dictionary with overall precision, recall, f1, accuracy
    """
    
    metric = evaluate.load("seqeval")

    # predictions -> logits, model predictions
    # true_labels -> ground truth
    predictions, true_labels = p
    # convert logits to predicted class ids
    predictions = np.argmax(predictions, axis=2)
    
    # convert input_ids to strings, ingoring padding tokens
    predictions_named = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, true_labels)
    ]
    true_labels_named = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, true_labels)
    ]
    
    # calculate metrics
    results = metric.compute(predictions=predictions_named, references=true_labels_named)
    
    results_overall = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }
    
    return results_overall
    
    

def compute_metrics_tok(p):
    """
    Computes precision, recall, F1 using the scikit-learn library.
    Token-based evaluation, counts predictions that match a token as correct predictions.
    
    Args:
        p: predictions and labels
    Returns:
        results_overall (dict): dictionary with overall precision, recall, f1, accuracy
    """
    
    # predictions -> logits, model predictions
    # true_labels -> ground truth
    predictions, true_labels = p
    # convert logits to predicted class ids
    predictions = np.argmax(predictions, axis=2)
    
    # flatten batches, filter special tokens
    predictions_flat = [
        p_val for prediction, label in zip(predictions, true_labels)
        for p_val, l_val in zip(prediction, label) if l_val != -100
    ]
    true_labels_flat = [
        l_val for prediction, label in zip(predictions, true_labels)
        for p_val, l_val in zip(prediction, label) if l_val != -100
    ]
    
    # calculate metrics
    # average='weighted' accounts for label imbalance (e.g. many 'O' labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels_flat, predictions_flat, average="weighted")

    accuracy = accuracy_score(true_labels_flat, predictions_flat)
    
    results_overall = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }
    
    print("\n--- Detailed scikit-learn classification report for token-based evaluation ---")
    print(classification_report(true_labels_flat, predictions_flat))
    
    return results_overall


# %%
# configuration
DATA_PATH = "./data/chia_without_scope_parsedNER_v1"
MODEL_CHECKPOINT = "emilyalsentzer/Bio_ClinicalBERT"
OUTPUT_DIR = "./models/ner_test"

# %%
# hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
WEIGHT_DECAY = 0.01



# %%
# main training loop
def main():
    # %%
    # for compute_metrics_seq function
    global id2label, label2id
    
    # %%
    # load dataset
    print(f"Loading dataset from {DATA_PATH}...")
    try:
        dataset = load_from_disk(DATA_PATH)
        print(f"Dataset loaded with splits: {dataset.keys()}")
    except FileNotFoundError:
        print("Error: dataset not found. Please run your preprocessing script first.")
        

    # %%
    # dataset splits
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    
    # %%
    # load label mappings
    label2id, id2label = load_label_map(DATA_PATH)
    label_list = list(label2id.keys())
    print(f"Loaded {len(label_list)} labels: {label_list[:5]}...")
    

    # %%
    # model initialization
    # resize classification head to match number of labels
    print("Initializing Model...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    
    
    # %%
    # data collator
    # automatically pads the inputs to the maximum length in the batch
    # (more efficient than padding everything to 512)
    # tokenizer for the data collator (to know how to pad)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    
    # %%
    # training arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",  
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        save_total_limit=2
    )
    
    
    # %%
    # initialize trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_seq,
    )
    
    
    # %%
    # start training
    print("Starting training...")
    trainer.train()
    
    
    # %%
    # final evaluation
    # on test dataset
    print("Evaluating final model...")
    metrics = trainer.evaluate(test_dataset)
    print("\nFinal Test Metrics:")
    print(metrics)
    
    # %%
    # saves model + tokenizer
    final_save_path = os.path.join(OUTPUT_DIR, "ner_chia_test")
    trainer.save_model(final_save_path)
    print(f"Model saved to {final_save_path}")
    

# %%
# boilerplate
if __name__ == "__main__":
    main()