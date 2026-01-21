
# to do
# - save ner results - eigene function
# - adjust ner results to jsonl, andere inhalte etc
# - save re results - eigene function
# - add full text/sentence to ner results



# %%
# imports
import os
import json
import torch
import itertools
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# %%
# functions
def run_ner_module(text, ner_pipe):
    """
    Applies NER model to text.
    Returns list of predicted entities.
    """
    # apply NER
    predictions = ner_pipe(text)
    
    # maybe --> filter low confidence entities
    # entities = [ent for ent in entities if ent['score'] > 0.4
    
    # format predictions for json serialization
    predicted_entities = list()
    for pred in predictions:
        clean_pred = {
            "entity_group": pred["entity_group"],
            "word": pred["word"],
            "start": int(pred["start"]),
            "end": int(pred["end"]),
            "score": float(pred["score"])
        }
        predicted_entities.append(clean_pred)
        
    return predicted_entities




def format_re_input(text, entity1, entity2):
    """
    Injects markers [E1]...[/E1] and [E2]...[/E2] for the RE model.
    Inserts from right-to-left to prevent index shifting.
    """
    e1_start, e1_end = entity1["start"], entity1["end"]
    e2_start, e2_end = entity2["start"], entity2["end"]
    insertions = [
        (e1_start, "[E1]"), (e1_end, "[/E1]"),
        (e2_start, "[E2]"), (e2_end, "[/E2]")
    ]
    
    # sort descending by position
    # --> to insert markers back to front
    insertions.sort(key=lambda x: x[0], reverse=True)

    marked_text = text
    for pos, marker in insertions:
        if pos > len(marked_text):
            continue
        marked_text = marked_text[:pos] + marker + marked_text[pos:]
    
    return marked_text



def run_re_module(text, entities, re_tokenizer, re_model):
    """
    Takes text and entities (from NER), generates pairs, and predicts relations.
    """
    
    if len(entities) < 2:
        print("Need at least 2 entities for relation extraction.")
        return []

    # generate permutations
    # entity pairs for possible relations
    pairs = list(itertools.permutations(entities, 2))
    found_relations = list()
    
    for e1, e2 in pairs:
        # skip overlapping entities
        if max(e1["start"], e2["start"]) < min(e1["end"], e2["end"]):
            continue
        
        # entity marker injection
        try:
            marked_text = format_re_input(text, e1, e2)
        except Exception as e:
            print(f"Warning: formatting error pair {e1["word"]} / {e2["word"]}: {e}")
            continue
        
        # tokenization
        inputs = re_tokenizer(
            marked_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length"
        )
        
        # relation prediction
        re_model.eval()
        with torch.no_grad():
            outputs = re_model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=1).item()
        pred_label = re_model.config.id2label[pred_id]
        confidence = torch.softmax(logits, dim=1)[0][pred_id].item()
        
        # save predicted relations
        if pred_label != "NO_RELATION":
            found_relations.append({
                "arg1": e1["word"],
                "arg1_type": e1["entity_group"],
                "arg2": e2["word"],
                "arg2_type": e2["entity_group"],
                "relation": pred_label,
                "confidence": confidence,
                "context": marked_text
            })
    
    return found_relations
    
    
    
# %%
if __name__ == "__main__":
    
    
    # %%
    # --- (0) configurations ---
    DATA_DIR = "./data_by_lea/data_parsed_sentLevel/ctg_sentLevel.csv"
    NER_MODEL_PATH = "./models/ner_chia_test2"
    RE_MODEL_PATH = "./models/re_test_small_fullDownsampled/checkpoint-52"
    OUTPUT_DIR = "./results/test"
    # create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    
    # %%
    # --- (1) load ctg-dataset ---
    print(f"\nLoading test dataset from {DATA_DIR}...")
    df = pd.read_csv(DATA_DIR)
    # test fragment
    df = df[:2]
    print(f"Test dataset successfully loaded.")
    
    
    # %%
    # --- (2) load NER model and tokenizer ---
    print("\nLoading NER model and tokenizer...")
    # - aggregation_strategy="simple"
    # --> merge sub-tokens into words
    # - device=-1
    # --> -1=CPU, 0=GPU
    ner_pipeline = pipeline(
        "token-classification",
        model=NER_MODEL_PATH,
        tokenizer=NER_MODEL_PATH,
        aggregation_strategy="simple",
        device=-1
    )
    print("NER model and tokenizer successfully loaded.")
    
    
    # %%
    # --- (3) load RE model and tokenizer ---
    print("\nLoading RE model and tokenizer...")
    re_tokenizer = AutoTokenizer.from_pretrained(RE_MODEL_PATH)
    re_model = AutoModelForSequenceClassification.from_pretrained(RE_MODEL_PATH)
    print("RE model and tokenizer successfully loaded.")
    
    
    # %%
    # --- (4) apply NER ---
    print(f"\nRunning NER inference on all sentences...")
    entities = list()
    entities_column = list()
    for idx, row in df.iterrows():
        text = row["Sentence"]
        ner_prediction = run_ner_module(text, ner_pipeline)
        sentence_entry = {
            "sentence_id": idx,
            "text": text,
            "entities": ner_prediction,
            "nct_id": row["StudyNCTid"],
            "criteria_type": row["criteria_type"]
        }
        entities.append(sentence_entry)
        # entities_column.append(json.dumps(ner_prediction))
        entities_column.append(ner_prediction)
    # save entities in dataframe
    df["Entities"] = entities_column
    print("NER inference on all sentences successfully completed.")
    
    
    # %%
    # --- (4.1) save NER predictions ---
    ner_output_path_jsonl = os.path.join(OUTPUT_DIR, "ner_predictions.jsonl")
    ner_output_path_csv = os.path.join(OUTPUT_DIR, "ner_predictions.csv")
    # JSONL format
    with open(ner_output_path_jsonl, "w", encoding="UTF-8") as f:
        for sent in entities:
            json.dump(sent, f)
            f.write("\n")
    print(f"\nNER predictions saved in JSONL-format to: {ner_output_path_jsonl}")
    # csv format
    df.to_csv(ner_output_path_csv, index=False)
    print(f"\nNER predictions saved in csv-format to: {ner_output_path_csv}")
    

    # %%
    # --- (5) apply RE to NER predictions ---
    relations = list()
    relations_column = list()
    for idx, row in df.iterrows():
        text = row["Sentence"]
        entities = row["Entities"]
        re_prediction = run_re_module(text, entities, re_tokenizer, re_model)
        sentence_entry = {
            "sentence_id": idx,
            "text": text,
            "relations": re_prediction,
            "nct_id": row["StudyNCTid"],
            "criteria_type": row["criteria_type"]
        }
        relations.append(sentence_entry)
        relations_column.append(re_prediction)

        # print()
        # print(f"Sentence: {text}")
        # print(f"Predicted relations:")
        # for rel in re_prediction:
        #     print(f"\t{rel["relation"]}({rel["arg1_type"]}({rel["arg1"]}), {rel["arg2_type"]}({rel["arg2"]}))")
    
    # save relations in dataframe
    df["Relations"] = relations_column
    print("RE inference on all sentences successfully completed.")
        
            
    # %%
    # --- (5.1) save RE results ---
    re_output_path_jsonl = os.path.join(OUTPUT_DIR, "re_predictions.jsonl")
    re_output_path_csv = os.path.join(OUTPUT_DIR, "re_predictions.csv")
    # JSONL format
    with open(re_output_path_jsonl, "w", encoding="UTF-8") as f:
        for sent in relations:
            json.dump(sent, f)
            f.write("\n")
    print(f"\nRE predictions saved in JSONL-format to: {re_output_path_jsonl}")
    # csv format
    df.to_csv(re_output_path_csv, index=False)
    print(f"\nRE predictions saved in csv-format to: {re_output_path_csv}")

    print(f"\nNER and RE inference pipeline successfully completed.")
    print(f"Results can be found at {OUTPUT_DIR}")