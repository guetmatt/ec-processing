# based on: https://huggingface.co/learn/llm-course/chapter7/2?fw=pt
# and: https://huggingface.co/docs/transformers/main/tasks/token_classification



# %%
# imports
from transformers import pipeline


# testdata
data = """
28:35:chronic_disease	has a significant risk for suicide
46:49:chronic_disease	has any current primary diagnosis other than mdd , where primary diagnosis is defined as the primary source of current distress and functional impairment
52:64:chronic_disease,67:78:chronic_disease,84:93:chronic_disease	has any other significant medical condition ( eg , neurological , psychiatric , or metabolic ) or clinical symptom that could unduly risk the subject or affect the interpretation of study data
14:49:treatment,61:79:upper_bound,102:105:treatment,128:155:treatment	has received electroconvulsive therapy treatment within the last @NUMBER years or within the current mde or failed a course of electroconvulsive treatment at any time
10:25:treatment,33:40:treatment,43:52:treatment,55:63:treatment,66:74:treatment,80:98:treatment,106:114:treatment,117:127:treatment,137:155:upper_bound	has used opioid agonists ( eg , codeine , oxycodone , tramadol , morphine ) or opioid antagonists ( eg , naloxone , naltrexone ) within @NUMBER days prior to screening
1:9:pregnancy	pregnant
1:20:chronic_disease,44:64:chronic_disease	auto-immune disease , acute stage ( e.g. , rheumatoid arthritis )
9:24:chronic_disease,27:49:chronic_disease,59:72:chronic_disease	certain mental diseases / psychiatric conditions ( e.g. , schizophrenia ) that would preclude reliable testing and participation
11:19:chronic_disease	diagnosed epilepsy
14:22:chronic_disease,55:73:chronic_disease	diagnosis of glaucoma ( not type-specific , excluding traumatic glaucoma ) : moderate defect or worse in both eyes but not total blindness
"""


# %%
data_lines = data.strip().split("\n")
data_processed = list()
for index, line in enumerate(data_lines):
    line_split = line.split("\t")
    data_processed.append(line_split[1])


# %%
model_checkpoint = "./ClinicalBERT_testing/checkpoint-1515"
ner_classifier = pipeline("ner", model=model_checkpoint, aggregation_strategy="simple")

# %%
test = ner_classifier(data_processed[2])
# %%
print(test)