# NEXT UP:
# - running inference on a single new sentence
# - switch from general bert to domain-specific bert


# train bert-based model for relation extraction
# with chia dataset (parsed and preprocessed)

# %%
# imports
import json
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# %%
# configuration
DATA_DIR = "./data/chia_tokenized_v1"
MODEL_OUTPUT_DIR = "./models/bert-re-v1"
BASE_MODEL = "bert-base-uncased" # must match tokenizer used earlier
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

# %%
# evaluation metrics
# MICRO or MACRO average to use?
def compute_metrics(eval_pred):
    """
    Calculates Precision, Recall, and F1-Score.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # calculate metrics
    # weighted -> accounts class imbalance
    # macro -> rare classes as important as common classes
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    
    acc = accuracy_score(labels, predictions)
    
    results = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
    
    return results

# %%
# training loop
def main():
    
    # load dataset
    print(f"Loading dataset from '{DATA_DIR}'...")
    dataset = load_from_disk(DATA_DIR)
    print(f"Dataset successfully loaded!")
    
    # label mapping
    with open(f"{DATA_DIR}/label_map.json", "r") as f:
        mapping = json.load(f)
        label2id = mapping["label2id"]
        # id2label -> store number-string as int
        id2label = {int(k): v for k, v in mapping["id2label"].items()}
        
    num_labels = len(label2id)
    print(f"Found {num_labels} classes: {list(label2id.keys())}")
    
    
    # load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(DATA_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # CRITICAL: resize embeddings to mach new tokenizer size
    # new tokens: [E1], etc.
    model.resize_token_embeddings(len(tokenizer))
    
    # training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
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
        logging_dir="./logs",
        logging_steps=100,
        report_to="none"
    )
    
    # trainer initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )
    
    # start training
    print("Training started...")
    trainer.train()
    print("Training finished.")
    
    # save trained model
    print(f"Saving best model to '{MODEL_OUTPUT_DIR}' (based on f1-score)")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    
    # model evaluation
    print("\n Final Evaluation Results:")
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))
    
# %%
if __name__ == "__main__":
    main()
    