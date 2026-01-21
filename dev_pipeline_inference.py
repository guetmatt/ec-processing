
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


# %%
# functions
def run_ner_module(text, ner_pipe):
    """
    Applies NER model to text.
    Returns list of entities and saves them to disk.
    """
    # apply NER
    entities = ner_pipe(text)
    
    # maybe --> filter low confidence entities
    # entities = [ent for ent in entities if ent['score'] > 0.4
    
    # save results
    ner_output_path = os.path.join(OUTPUT_DIR, "ner_output.json")
    with open(ner_output_path, "w", encoding="UTF-8") as f:
        serializable_entities = list()
        for ent in entities:
            ent_clean = ent.copy()
            ent_clean["score"] = float(ent["score"])
            ent_clean["start"] = int(ent["start"])
            ent_clean["end"] = int(ent["end"])
            serializable_entities.append(ent_clean)
        json.dump(serializable_entities, f, indent=4)
    print(f"NER results saved to {ner_output_path}")
    
    return serializable_entities



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
    
    
    
    

def main():
    # (1) load NER model and tokenizer
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
    
    # (2) load RE model and tokenizer
    print("\nLoading RE model and tokenizer...")
    re_tokenizer = AutoTokenizer.from_pretrained(RE_MODEL_PATH)
    re_model = AutoModelForSequenceClassification.from_pretrained(RE_MODEL_PATH)
    print("RE model and tokenizer successfully loaded.")

    # (3) input
    example_text = "Pregnant females."
    
    # (4) apply NER
    entities = run_ner_module(example_text, ner_pipeline)
    
    # (5) apply RE to NER results   
    relations = run_re_module(example_text, entities, re_tokenizer, re_model)
    if relations:
        print(f"RE identified {len(relations)} relations:")
        for rel in relations:
            print(f"\tEnt1: {rel["arg1"]}, Ent2: {rel["arg2"]}, Relation: [{rel["relation"]}], Confidence: ({rel["confidence"]:.2f})")
    else:
        print("RE found no relations.")
    
    
    # (6) save RE results 
    re_output_path = os.path.join(OUTPUT_DIR, "re_output.json")
    with open(re_output_path, "w", encoding="UTF-8") as f:
        json.dump(relations, f, indent=4)
    
    print(f"\nFinal relation extraction results saved to {re_output_path}")
    

# %%
if __name__ == "__main__":
    
    # %%
    # configurations
    NER_MODEL_PATH = "./models/ner_chia_test2"
    RE_MODEL_PATH = "./models/re_test_small_fullDownsampled/checkpoint-52"
    OUTPUT_DIR = "./results/test"
    # create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # %%
    main()