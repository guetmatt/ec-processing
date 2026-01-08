# 3. Next Step: Batch Processing for CTG For the actual CTG dataset processing, do not loop one by one as shown in the "TEST" block. Instead:
# Run NER on the whole CTG dataset (batches).
# Store all extracted entities in a JSON file (intermediate storage).
# Load the JSON, generate pairs, and run RE in batches.


# TO DO
# - store NER results as file
# - load NER result-files for RE
# - store RE results as file
# - work on RE trainng
# - work on NER training
 


# %%
# imports
import torch
import itertools
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification


# %%
# functions
def extract_entities_from_ner(text, tokenizer, model):
    """
    Run NER, aggregate BIO tags to character-level entity spans for RE.
    Returns:
        - list of dicts {"entity": label, "start": char_idx, "end": char_idx, "text": str}
    """

    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs["offset_mapping"][0].tolist()
    
    with torch.no_grad():
        logits = model(**{k: v for k, v in inputs.items() if k != "offset_mapping"}).logits

    predictions = torch.argmax(logits, dim=2)[0].tolist()
    id2label = model.config.id2label
    
    entities = list()
    current_entity = None
    
    for idx, pred_id in enumerate(predictions):
        label = id2label[pred_id]
        start_char, end_char = offset_mapping[idx]
        
        # ignore special tokens ([CLS], [PAD], etc.)
        if start_char == end_char == 0:
            continue
        
        # beginning of an entity
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "label": label[2:], # remove "B-"
                "start": start_char,
                "end": end_char,
                "tokens": [idx]
            }
        # inside of an entity
        elif label.startswith("I-") and current_entity:
            # label consistent with previous entity label
            if label[2:] == current_entity["label"]:
                current_entity["end"] = end_char
                current_entity["tokens"].append(idx)
            # label not consistent with previous entity label
            else:
                entities.append(current_entity)
                current_entity = None
        
        # label: no entity (after last token of current entity)
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        
    if current_entity:
        entities.append(current_entity)
    
    # extract text of entities
    # for verification
    for ent in entities:
        ent["text"] = text[ent["start"]:ent["end"]]
    
    return entities



def format_re_input(text, entity_1, entity_2):
    """
    Injects [E1]...[/E1] and [E2]...[/E2] markers into the text.
    Handles overlapping entities by prioritizing strict injection order.
    """
    
    # sort by start index
    # -> handle insertion correctly
    # insert the later entity first
    # -> indices of earlier entity dont shift
    e1_start, e1_end = entity_1["start"], entity_1["end"]
    e2_start, e2_end = entity_2["start"], entity_2["end"]
    
    insertions = [
        (e1_start, "[E1]"), (e1_end, "[/E1]"),
        (e2_start, "[E2]"), (e2_end, "[/E2]")
    ]
    
    # reverse = position descending (to insert later entities first)
    insertions.sort(key=lambda x: x[0], reverse=True)
    
    
    marked_text = text
    for pos, marker in insertions:
        marked_text = marked_text[:pos] + marker + marked_text[pos:]
        
    return marked_text



def predict_relation(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    
    return model.config.id2label[pred_id]



# main pipeline
def process_sentence(sentence):
    print(f"\nOriginal sentence: {sentence}")
    
    # NER
    entities = extract_entities_from_ner(sentence, ner_tokenizer, ner_model)
    entities_and_labels = [ent["text"] + " (" + ent["label"] + ")" for ent in entities]
    print(f"Found {len(entities)} entities: {entities_and_labels}")
    
    if len(entities) < 2:
        print("Not enough entities for relation extraction. Need at least two entities.")
        return None
    
    # candidate pair generation for RE
    # permutations -> implies directionality of relations matters
    pairs = list(itertools.permutations(entities, 2))
    
    print(f"Testing {len(pairs)} candidate pairs for RE...")
    for e1, e2 in pairs:
        # maker injection
        try:
            marked_text = format_re_input(sentence, e1, e2)
        # skip overlapping entities
        except Exception as e:
            print(f"Skipping pair {e1, e2} due to overlap/error: {e}")
            continue
        
        # RE
        relation = predict_relation(marked_text, re_tokenizer, re_model)
        
        # relation found
        if relation != "NO_RELATION":
            print(f"RELATION FOUND: {e1["text"]} -> {relation} -> {e2["text"]}") 
            print(f"\tContext: {marked_text}")
        else:
            print(f"No relation found.")
    
           

# %%
# testing
if __name__ == "__main__":
    
    # configuration
    NER_MODEL_PATH = "./models/ner_chia_test2"
    RE_MODEL_PATH = "./models/bert-re-v1"
    MAX_LEN = 256

    # %%
    # load models
    print("Loading models...")

    ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
    ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)

    re_tokenizer = AutoTokenizer.from_pretrained(RE_MODEL_PATH)
    re_model = AutoModelForSequenceClassification.from_pretrained(RE_MODEL_PATH)

    # %%
    # models in eval-mode (ignores some training-specific layers)
    ner_model.eval()
    re_model.eval()
    
    # %%
    # example sentence
    test_sentence = "Pregnant female."
    process_sentence(test_sentence)