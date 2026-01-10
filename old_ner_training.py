""" 
NEXT UP:
- SET UP MY OWN DATA TO CLASSIFY ! (DATASET FROM LEA)
- TERM NORMALIZATION??? CMAYBE JUST USE AN ALREADY TRAINED MODEL?
- RELATION EXTRACTION TASK

"""



# based on: https://huggingface.co/learn/llm-course/chapter7/2?fw=pt
# and: https://huggingface.co/docs/transformers/main/tasks/token_classification

# training ner-task with huggingface.transformers functions

# %%
# imports internal
from preprocess_ctpDataset import *

# imports external
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification
import evaluate
import numpy as np


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    # Remove ignored index (special tokens) and convert to labels
    true_labels = list()
    for label in labels:
        sent_labels = list()
        for l in label:
            if l != -100:
                sent_labels.append(id2label[l])
        true_labels.append(sent_labels)
    true_predictions = list()
    for prediction, label in zip(predictions, labels):
        sent_predictions = list()
        for (p, l) in zip(prediction, label):
            if l != -100:
                sent_predictions.append(id2label[p])
        true_predictions.append(sent_predictions)
                
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }




# boilerplate
if __name__ == "__main__":
    
    # choose model
    model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # set up evaluation metric
    metric = evaluate.load("seqeval")

    # load dataset
    dataset, id2label, label2id = preprocess_ctp("./data_ner_clinicalTrialParser/test_processed_medical_ner.tsv", tokenizer_model=model_checkpoint)
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    dataset_train = splits["train"]
    dataset_eval = splits["test"]


    # data collator for padding
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    # batch = data_collator([dataset[i] for i in range(len(dataset))])

    # model
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id
    )
    
    # REFINE DISK-SAVING OF TRAINED MODEL!
    # define training arguments
    args = TrainingArguments(
        output_dir="ClinicalBERT_testing",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False
        )
    
    # training class
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer
    )
    trainer.train()
    
    
