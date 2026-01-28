# TO DECIDE
# - compute_metrics_tok or compute_metrics_seq ???
# - handle this warning during training loop:
#    /usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565:
#       UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
#       Use `zero_division` parameter to control this behavior.
#   _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))


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
import optuna
import sys
from optuna.visualization import plot_param_importances, plot_parallel_coordinate


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
    
    # classification report
    # with label_names instead of label_ids
    global id2label
    sorted_ids = sorted_ids = sorted(id2label.keys())
    sorted_names = [id2label[id] for id in sorted_ids]
    print("\n--- Detailed scikit-learn classification report for token-based evaluation ---")
    print(classification_report(
        true_labels_flat,
        predictions_flat,
        labels=sorted_ids,
        target_names=sorted_names,
        zero_division=0
    ))

    return results_overall



def model_init(trial=None):
    """
    Re-initializes the model for every trial of hyperparameter-optimization.
    The 'trial' argument is required by the Trainer but is not used here.
    """
    
    label2id, id2label = load_label_map(DATA_PATH)
    
    return AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )



def hyperparameter_space(trial):
    """
    Defines the hyperparameter search space.
    Here: stick to standard narrow ranges.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.3),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 4, 8)
    }



# %%
# main training loop
def main():
    # %%
    # for compute_metrics_seq function
    global id2label, label2id
    global DATA_PATH, MODEL_CHECKPOINT
    
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
    

    # WITHOUT HYPERPARAMETER OPTIMIZATION
    # # %%
    # # model initialization
    # # resize classification head to match number of labels
    # print("Initializing Model...")
    # model = AutoModelForTokenClassification.from_pretrained(
    #     MODEL_CHECKPOINT,
    #     num_labels=len(label_list),
    #     id2label=id2label,
    #     label2id=label2id
    # )
    
    
    # %%
    # data collator
    # automatically pads the inputs to the maximum length in the batch
    # (more efficient than padding everything to 512)
    # tokenizer for the data collator (to know how to pad)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    
    # WITHOUT HYPERPARAMETER OPTIMIZATION
    # # %%
    # # training arguments
    # args = TrainingArguments(
    #     output_dir=OUTPUT_DIR,
    #     eval_strategy="epoch",
    #     save_strategy="epoch",  
    #     learning_rate=LEARNING_RATE,
    #     per_device_train_batch_size=BATCH_SIZE,
    #     per_device_eval_batch_size=BATCH_SIZE,
    #     num_train_epochs=NUM_EPOCHS,
    #     weight_decay=WEIGHT_DECAY,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="f1",
    #     logging_dir=f"{OUTPUT_DIR}/logs",
    #     logging_steps=50,
    #     save_total_limit=2,
    #     # colab-specific settings
    #     fp16 = True, # enhances training speed on gpu
    #     report_to="none", # disables WandB prompts
    #     dataloader_num_workers=2, # speeds up data loading
    # )
    
    # WTIH HYPERPARAMETER OPTIMIZATION
    # %%
    # training arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        fp16=True,
        disable_tqdm=False,
        report_to="none"
    )
    
    
    # WITHOUT HYPERPARAMETER OPTIMIZATION
    # # %%
    # # initialize trainer
    # trainer = Trainer(
    #     model=model,
    #     args=args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics_tok,
    # )
    
    # WITH HYPERPARAMETER OPTIMIZATION
    # %%
    # initialize trainer
    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_tok
    )
    
    
    # %%
    # Hyperparameter optimization search
    print(f"Starting hyperparameter search...")
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=hyperparameter_space,
        backend="optuna",
        n_trials=2, # 10 for testing, 20+ for final model
        study_name="ner_hpo_search"
    )
    print(f"Best run found: {best_run}")
    
    # retrain on full parameters of best_run
    # update args with best_run parameters
    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)
    
    
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
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    

# %%
# boilerplate
if __name__ == "__main__":
    
    # %%
    # local paths
    DATA_PATH = "./data/chia_without_scope_parsedNER_lines_full_100126"
    MODEL_CHECKPOINT = "emilyalsentzer/Bio_ClinicalBERT"
    OUTPUT_DIR = "./models/NER_chia_lines_test_hpo"

    # # %%
    # # paths for google colab
    # BASE_PATH = "/content/drive/MyDrive/masters_thesis_computing"
    # DATA_PATH = os.path.join(BASE_PATH, "data/chia_without_scope_parsedNER_lines_full_100126")
    # print(f"DATA_PATH = {DATA_PATH}")
    # OUTPUT_DIR = os.path.join(BASE_PATH, "models/NER_chia_lines_full_100126")
    # MODEL_CHECKPOINT = "emilyalsentzer/Bio_ClinicalBERT"

    # %%
    # hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    WEIGHT_DECAY = 0.01
    
    # %%
    main()
    
    
    
    
    # # GETTING SOME DATA STATS
    
    # # %%
    # # load dataset
    # print(f"Loading dataset from {DATA_PATH}...")
    # try:
    #     dataset = load_from_disk(DATA_PATH)
    #     print(f"Dataset loaded with splits: {dataset.keys()}")
    # except FileNotFoundError:
    #     print("Error: dataset not found. Please run your preprocessing script first.")
        

    # # %%
    # # dataset splits
    # train_dataset = dataset["train"]
    # eval_dataset = dataset["validation"]
    # test_dataset = dataset["test"]
    
    # # %%
    # # load label mappings
    # label2id, id2label = load_label_map(DATA_PATH)
    # label_list = list(label2id.keys())
    # print(f"Loaded {len(label_list)} labels: {label_list[:5]}...")
    
    
    # # %%
    # print(train_dataset.features)
    # print(train_dataset[0]["labels"])
    # #%%
    # counter = 0
    # for idx, entry in enumerate(train_dataset):
    #     labels = list()
    #     for label_id in entry["labels"]:
    #         labels.append(id2label[label_id])
        
    #     if "B-Subjective_judgement" in labels:
    #         print(entry["sentence_text"])
    #         counter += 1
        
    # print(counter)    