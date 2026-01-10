# # based on: https://huggingface.co/learn/llm-course/chapter7/2?fw=pt
# # and: https://huggingface.co/docs/transformers/main/tasks/token_classification


# # %%
# from transformers import AutoTokenizer
# from datasets import Dataset
# from transformers import DataCollatorForTokenClassification
# import evaluate
# import numpy as np
# from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification
# from accelerate import Accelerator
# from torch.utils.data import DataLoader
# from torch.optim import AdamW
# from transformers import get_scheduler
# from tqdm.auto import tqdm
# import torch


# data = """
# 28:35:chronic_disease	has a significant risk for suicide
# 46:49:chronic_disease	has any current primary diagnosis other than mdd , where primary diagnosis is defined as the primary source of current distress and functional impairment
# 52:64:chronic_disease,67:78:chronic_disease,84:93:chronic_disease	has any other significant medical condition ( eg , neurological , psychiatric , or metabolic ) or clinical symptom that could unduly risk the subject or affect the interpretation of study data
# 14:49:treatment,61:79:upper_bound,102:105:treatment,128:155:treatment	has received electroconvulsive therapy treatment within the last @NUMBER years or within the current mde or failed a course of electroconvulsive treatment at any time
# 10:25:treatment,33:40:treatment,43:52:treatment,55:63:treatment,66:74:treatment,80:98:treatment,106:114:treatment,117:127:treatment,137:155:upper_bound	has used opioid agonists ( eg , codeine , oxycodone , tramadol , morphine ) or opioid antagonists ( eg , naloxone , naltrexone ) within @NUMBER days prior to screening
# 1:9:pregnancy	pregnant
# 9:24:chronic_disease,27:49:chronic_disease,59:72:chronic_disease	certain mental diseases / psychiatric conditions ( e.g. , schizophrenia ) that would preclude reliable testing and participation
# 14:22:chronic_disease,55:73:chronic_disease	diagnosis of glaucoma ( not type-specific , excluding traumatic glaucoma ) : moderate defect or worse in both eyes but not total blindness
# 1:24:chronic_disease,83:91:chronic_disease,94:100:chronic_disease	end-stage organ disease or medical condition with subsequent vision loss ( e.g. , diabetes , stroke )
# 1:18:chronic_disease,30:48:upper_bound	epileptic seizure within the past @NUMBER years of enrollment date
# 1:29:clinical_variable,32:44:lower_bound	intraocular pressure ( iop ) > @NUMBER mmhg at baseline
# 21:36:chronic_disease,64:106:clinical_variable,109:116:upper_bound	medically diagnosed memory disorder or telephone interview for cognitive status-modified ( tics-m ) score ≤ @NUMBER
# 1:30:treatment	metallic artifacts / implants in head and / or torso
# 33:42:chronic_disease,70:87:lower_bound	other diseases of the retina or cataracts responsible for worse than @NUMBER / @NUMBER best-corrected visual acuity
# 30:38:chronic_disease	other optic comorbidity than glaucoma
# 10:28:chronic_disease,38:46:chronic_disease,66:86:chronic_disease	unstable medical conditions ( e.g. , diabetes , diabetes causing diabetic retinopathy )
# 1:21:chronic_disease,43:57:lower_bound	visual field defects present for at least @NUMBER months
# 24:38:treatment,44:57:lower_bound	anticipated to undergo pancreatectomy in ≥ @NUMBER weeks from enrollment
# 11:31:treatment	completed preoperative therapy and are on their presurgical rest period
# 1:10:chronic_disease,14:35:chronic_disease	myopathic or rheumatologic disease that impacts physical function
# 1:22:cancer	neuroendocrine cancer
# 1:26:clinical_variable,32:39:lower_bound,47:54:upper_bound	numeric pain rating scale of ≥ @NUMBER out of @NUMBER
# 1:18:cancer	pancreatic cancer of any type , biopsy-proven
# 1:4:age,7:20:upper_bound	age < @NUMBER years
# 1:18:treatment	bariatric surgery patients
# 1:38:treatment	laparoscopic roux-en-y gastric bypass
# 27:47:treatment	patients undergoing other bariatric procedures
# 1:32:treatment	pre-operative opioid analgesics
# 21:45:treatment	previous history of roux-en-y gastric bypass"""


# data_lines = data.strip().split("\n")



# # create dataset as dictionary
# # example entry:
# # {
# #   'text': 'has a significant risk for suicide',
# #   'entities': 
# #       [{
# #       'start': 27,
# #       'end': 35,
# #       'label': 'chronic_disease',
# #       'entity_text': 'suicide'
# #       }]
# #   }
# dataset = dict()
# for index, line in enumerate(data_lines):
#     line_split = line.split("\t")
#     text = line_split[1]
    
#     dataset[index] = {
#         "text": text,
#         "entities": list()
#     }
    
#     tags = line_split[0].split(",")
#     for index2, tag in enumerate(tags):
#         tag_split = tag.split(":")
        
#         if len(tag_split) % 3 != 0:
#             print(f"Error in line {index}: {line}")
#             break
            
#         dataset[index]["entities"].append(
#             {
#                 "start": int(tag_split[0]) - 1, # -1, original data indexing starts at 1 
#                 "end": int(tag_split[1]),
#                 "label": tag_split[2],
#                 "entity_text": text[int(tag_split[0])-1:int(tag_split[1])]
#             }
#         )


# # %%

# # restructure dataset to NER-BIO scheme
# # example entry
# #{
# #   'text': ['has', 'a', 'significant', 'risk', 'for', 'suicide'],
# #   'labels': ['O', 'O', 'O', 'O', 'O', 'B-chronic_disease']
# # }
# dataset_new = dict()
# ner_labels = set()
# for key, val in dataset.items():
#     text = val["text"]
#     entities = val["entities"]
    
#     tokens = list()
#     token_spans = list()
    
#     # tokenize and keep character spans
#     pos = 0
#     for tok in text.split():
#         start = text.index(tok, pos)
#         end = start + len(tok)
#         tokens.append(tok)
#         token_spans.append((start, end))
#         pos = end
    
#     # initiate list of NER-labels corresponding to tokens
#     # "O" will be replaced if entity 
#     labels = ["O"] * len(tokens)
    
#     # assign NER labels in BIO scheme
#     for ent in entities:
#         # tags from original dataset
#         ent_start = ent["start"]
#         ent_end = ent["end"]
#         ent_label = ent["label"]
        
#         # if token span is in entity span from original dataset
#         # --> NER tag
#         first = True
#         for index, (tok_start, tok_end) in enumerate(token_spans):
#             if tok_start >= ent_start and tok_end <= ent_end:
#                 if first:
#                     labels[index] = f"B-{ent_label}"
#                     first = False
#                 else:
#                     labels[index] = f"I-{ent_label}"
    
#     dataset_new[key] ={
#         "text": tokens,
#         "labels": labels
#     } 
#     ner_labels = ner_labels.union(set(labels))


# # %%
# # print(dataset_new)
# # print(dataset_new[3]["text"])
# # print(dataset_new[3]["labels"])

# # restructure dataset-dictionary to dataset-class from huggingface transformers
# text_list = list()
# labels_list = list()
# for key, val in dataset_new.items():
#     text_list.append(val["text"])
#     labels_list.append(val["labels"])
# dataset_restructured = {
#     "text": text_list,
#     "labels": labels_list
# }

# ds = Dataset.from_dict(dataset_restructured)
# # print(ds)
# # print(ds["text"])
# # print(ds["labels"])
# # print(ds[0])
# # print(ds[0]["text"])

# # %%
# # model: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
# model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# # # %%
# # # TEST SAMPLE - HOW DOES THE TOKENIZER WORK?
# # # tokenize dataset with model-tokenizer
# # ds_tok = tokenizer(ds[0]["text"], is_split_into_words=True)
# # print(ds_tok)
# # tokens = tokenizer.convert_ids_to_tokens(ds_tok["input_ids"])
# # print(tokens)


# # %%
# # tokenize dataset with model-tokenizer and
# # align (sub-)tokens with NER-labels
# # (subwords, special tokens etc)
# def align_tokens_with_labels(examples):
#     tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=True)
    
#     # print(examples["labels"])
#     # print(tokenized_inputs.word_ids(batch_index=1))
#     # print(examples["labels"][1])
#     # print(tokenized_inputs["input_ids"][1])
#     # print(tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][1]))
    
#     labels = []
#     for index, label in enumerate(examples[f"labels"]):
#         word_ids = tokenized_inputs.word_ids(batch_index=index)  # Map tokens to their respective word.
#         previous_word_idx = None
#         label_ids = []
#         for word_idx in word_ids:  # Set the special tokens to -100.
#             if word_idx is None:
#                 label_ids.append("-100")
#             elif word_idx != previous_word_idx:  # Only label the first token of a given word.
#                 label_ids.append(label[word_idx])
#             else:
#                 label_ids.append("-100")
#             previous_word_idx = word_idx

#         labels.append(label_ids)

#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs
   
   
# # %%
# ds_tokenized = ds.map(align_tokens_with_labels, batched=True)
# # tokens = tokenizer.convert_ids_to_tokens(ds_tokenized[-1]["input_ids"])
# # print(tokens)

# # # %%
# # print(ds_tokenized[-1]["labels"])

# # %%
# # create set of used ner labels
# ner_labels = set()
# for entry in ds_tokenized:
#     ner_labels = ner_labels.union(set(entry["labels"]))

# # %%
# # create mapping dictionaries 
# # labels <-> label_ids
# def label_id_mapping(ner_labels: set):
#     labels = sorted(list(ner_labels), reverse=True) # reverse -> "O" in front = 0
#     labels.remove("-100")
#     id2label = dict()
#     label2id = dict()
#     for index, label in enumerate(labels):
#         id2label[index] = label
#         label2id[label] = index
#     id2label[-100] = "-100"
#     label2id["-100"] = -100
    
#     return id2label, label2id

# # %%
# id2label, label2id = label_id_mapping(ner_labels)


# # %%
# # add label_ids as column to dataset
# # corresponding to labels
# label_ids = list()
# for entry in ds_tokenized:
#     labels_as_ids = list()
#     for label in entry["labels"]:
#         labels_as_ids.append(label2id[label])
#     label_ids.append(labels_as_ids)
# ds_tokenized = ds_tokenized.add_column("label_ids", label_ids)


# # %%
# # preparing dataset for training
# data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
# ds_tok = ds_tokenized.remove_columns("labels")
# ds_tok = ds_tok.remove_columns("text")
# ds_tok = ds_tok.rename_column("label_ids", "labels")
# # padding
# batch = data_collator([ds_tok[i] for i in range(len(ds_tok))])


# # %%
# # metrics for (in-training) evaluation
# metric = evaluate.load("seqeval")

# # # %%
# # # testing
# # labels = ds_tok[0]["labels"]
# # labels = [id2label[id] for id in labels]
# # labels

# # # %%
# # # testing fake predictions
# # pred = labels.copy()
# # pred[2] = "B-chronic_disease"
# # metric.compute(predictions=[pred], references=[labels])

# # %%
# def compute_metrics(eval_preds):
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
    
#     # Remove ignored index (special tokens) and convert to labels
#     true_labels = list()
#     for label in labels:
#         sent_labels = list()
#         for l in label:
#             if l != -100:
#                 sent_labels.append(id2label[l])
#         true_labels.append(sent_labels)
#     true_predictions = list()
#     for prediction, label in zip(predictions, labels):
#         sent_predictions = list()
#         for (p, l) in zip(prediction, label):
#             if l != -100:
#                 sent_predictions.append(id2label[l])
#         true_predictions.append(sent_predictions)
                
#     all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
#     return {
#         "precision": all_metrics["overall_precision"],
#         "recall": all_metrics["overall_recall"],
#         "f1": all_metrics["overall_f1"],
#         "accuracy": all_metrics["overall_accuracy"],
#     }
    
# # %%    
# # model
# # EVTL MUSS ICH "-100" NOCH AUS LABELS RAUSNEHMEN
# model = AutoModelForTokenClassification.from_pretrained(
#     model_checkpoint,
#     id2label=id2label,
#     label2id=label2id
# )


# # %%
# # define training arguments
# args = TrainingArguments(
#     output_dir="ClinicalBERT_testing",
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=2e-5,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     push_to_hub=False
# )



# # %%
# # training
# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=ds_tok,
#     eval_dataset=ds_tok,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     processing_class=tokenizer
# )
# trainer.train()



# %%
# full training loop
# DEFINE TRAINING AND VALIDATION DATASET!
train_dataloader = DataLoader(
    ds_tok,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8
)

eval_dataloader = DataLoader(
    ds_tok,
    collate_fn=data_collator,
    batch_size=8
)

optimizer = AdamW(model.parameters(), lr=2e-5)

accelerator = Accelerator()

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)


num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)


output_dir = "ClinicalBERT_testing"


# %%

def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()
    
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


# %%
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    
    # training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    
    # evaluation
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        
        # Necessary to pad predictions and labels for being gathered
        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
        
        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)
        
        # true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
        true_labels, true_predictions = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=true_predictions, references=true_labels)
    
    results = metric.compute()
    print(
        f"epoch {epoch}:",
        {
            key: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )
    
    # Save
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
