# based on: https://huggingface.co/learn/llm-course/chapter7/2?fw=pt
# and: https://huggingface.co/docs/transformers/main/tasks/token_classification


# %%
from transformers import AutoTokenizer
import evaluate
import numpy as np


#%%
data = """
28:35:chronic_disease	has a significant risk for suicide
46:49:chronic_disease	has any current primary diagnosis other than mdd , where primary diagnosis is defined as the primary source of current distress and functional impairment
52:64:chronic_disease,67:78:chronic_disease,84:93:chronic_disease	has any other significant medical condition ( eg , neurological , psychiatric , or metabolic ) or clinical symptom that could unduly risk the subject or affect the interpretation of study data
14:49:treatment,61:79:upper_bound,102:105:treatment,128:155:treatment	has received electroconvulsive therapy treatment within the last @NUMBER years or within the current mde or failed a course of electroconvulsive treatment at any time
10:25:treatment,33:40:treatment,43:52:treatment,55:63:treatment,66:74:treatment,80:98:treatment,106:114:treatment,117:127:treatment,137:155:upper_bound	has used opioid agonists ( eg , codeine , oxycodone , tramadol , morphine ) or opioid antagonists ( eg , naloxone , naltrexone ) within @NUMBER days prior to screening
1:9:pregnancy	pregnant
9:24:chronic_disease,27:49:chronic_disease,59:72:chronic_disease	certain mental diseases / psychiatric conditions ( e.g. , schizophrenia ) that would preclude reliable testing and participation
14:22:chronic_disease,55:73:chronic_disease	diagnosis of glaucoma ( not type-specific , excluding traumatic glaucoma ) : moderate defect or worse in both eyes but not total blindness
1:24:chronic_disease,83:91:chronic_disease,94:100:chronic_disease	end-stage organ disease or medical condition with subsequent vision loss ( e.g. , diabetes , stroke )
1:18:chronic_disease,30:48:upper_bound	epileptic seizure within the past @NUMBER years of enrollment date
1:29:clinical_variable,32:44:lower_bound	intraocular pressure ( iop ) > @NUMBER mmhg at baseline
21:36:chronic_disease,64:106:clinical_variable,109:116:upper_bound	medically diagnosed memory disorder or telephone interview for cognitive status-modified ( tics-m ) score ≤ @NUMBER
1:30:treatment	metallic artifacts / implants in head and / or torso
33:42:chronic_disease,70:87:lower_bound	other diseases of the retina or cataracts responsible for worse than @NUMBER / @NUMBER best-corrected visual acuity
30:38:chronic_disease	other optic comorbidity than glaucoma
10:28:chronic_disease,38:46:chronic_disease,66:86:chronic_disease	unstable medical conditions ( e.g. , diabetes , diabetes causing diabetic retinopathy )
1:21:chronic_disease,43:57:lower_bound	visual field defects present for at least @NUMBER months
24:38:treatment,44:57:lower_bound	anticipated to undergo pancreatectomy in ≥ @NUMBER weeks from enrollment
11:31:treatment	completed preoperative therapy and are on their presurgical rest period
1:10:chronic_disease,14:35:chronic_disease	myopathic or rheumatologic disease that impacts physical function
1:22:cancer	neuroendocrine cancer
1:26:clinical_variable,32:39:lower_bound,47:54:upper_bound	numeric pain rating scale of ≥ @NUMBER out of @NUMBER
1:18:cancer	pancreatic cancer of any type , biopsy-proven
1:4:age,7:20:upper_bound	age < @NUMBER years
1:18:treatment	bariatric surgery patients
1:38:treatment	laparoscopic roux-en-y gastric bypass
27:47:treatment	patients undergoing other bariatric procedures
1:32:treatment	pre-operative opioid analgesics
21:45:treatment	previous history of roux-en-y gastric bypass"""


#%%
data_lines = data.strip().split("\n")

#%%
dataset = dict()
for index, line in enumerate(data_lines):
    line_split = line.split("\t")
    text = line_split[1]
    
    dataset[index] = {
        "text": text,
        "entities": list()
    }
    
    tags = line_split[0].split(",")
    for index2, tag in enumerate(tags):
        tag_split = tag.split(":")
        
        if len(tag_split) % 3 != 0:
            print(f"Error in line {index}: {line}")
            break
            
        dataset[index]["entities"].append(
            {
                "start": int(tag_split[0]) - 1, # -1, original data indexing starts at 1 
                "end": int(tag_split[1]),
                "label": tag_split[2],
                "entity_text": text[int(tag_split[0])-1:int(tag_split[1])]
            }
        )

# example entry of dataset dictionary
# {
#   'text': 'has a significant risk for suicide',
#   'entities': 
#       [{
#       'start': 27,
#       'end': 35,
#       'label': 'chronic_disease',
#       'entity_text': 'suicide'
#       }]
#   }

                    
#%%
for el in dataset:
    print(dataset[el])


# restructure dataset to NER-BIO scheme
# %%
dataset_new = dict()
ner_labels = set()

for k, v in dataset.items():
    text = v["text"]
    entities = v["entities"]
    
    tokens = list()
    token_spans = list()
    
    # tokenize and keep character spans
    pos = 0
    for tok in text.split():
        start = text.index(tok, pos)
        end = start + len(tok)
        tokens.append(tok)
        token_spans.append((start, end))
        pos = end
    
    # initiate list of NER-labels corresponding to tokens
    # "O" will be replaced if entity 
    labels = ["O"] * len(tokens)
    
    # assign NER labels in BIO scheme
    for ent in entities:
        # tags from original dataset
        ent_start = ent["start"]
        ent_end = ent["end"]
        ent_label = ent["label"]
        
        # if token span is in entity span from original dataset
        # --> NER tag
        first = True
        for index, (tok_start, tok_end) in enumerate(token_spans):
            if tok_start >= ent_start and tok_end <= ent_end:
                if first:
                    labels[index] = f"B-{ent_label}"
                    first = False
                else:
                    labels[index] = f"I-{ent_label}"
    
    dataset_new[k] ={
        "text": tokens,
        "labels": labels
    } 
    ner_labels = ner_labels.union(set(labels))

# %%
ner_labels = sorted(list(ner_labels))
id2label = dict()
label2id = dict()  
for index, label in enumerate(ner_labels):
    id2label[index] = label
    label2id[label] = index

# %%
print(dataset_new)
print(ner_labels)
print(id2label)
print(label2id)
# %%
print(dataset_new[0]["text"])
print(dataset_new[0]["labels"])



# %%
# model: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# %%
# align ClinicalBERT tokenization with NER-labels
# (subwords, special tokens etc)
def align_tokens_with_labels(tokenized_data, data_line):
    
    # for index, label in enumerate(data_dict[3]["labels"]): # [3] example data
    word_ids = tokenized_data.word_ids()
    previous_word_id = None
    label_ids = list()
        
    for word_id in word_ids:
        if word_id is None:
            label_ids.append(-100)
        # Only label the first (sub)token of a given word
        # https://datascience.stackexchange.com/questions/69640/what-should-be-the-labels-for-subword-tokens-in-bert-for-ner-task
        elif word_id != previous_word_id:
            label_ids.append(data_line["labels"][word_id])
        else:
            label_ids.append(-100)
        previous_word_id = word_id

    
    tokenized_data["labels"] = label_ids    
    
    return tokenized_data


# %%
# align dataset after tokenizationwith ClinicalBERT
dataset_aligned = dict()
for index in dataset_new:
    # tokenize line with ClinicalBERT
    tokens = tokenizer(dataset_new[index]["text"], is_split_into_words=True)
    tokens = align_tokens_with_labels(tokens, dataset_new[index])

    dataset_aligned[index] = tokens

print(dataset_aligned[0].word_ids())
print(dataset_aligned[0].tokens())
print(dataset_aligned[0].labels)
