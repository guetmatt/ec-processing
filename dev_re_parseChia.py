# TO DO
# - parse whole dataset



# %%
# imports
import os
import glob
import re
import pandas as pd
import itertools
import json
from datasets import Dataset, DatasetDict, ClassLabel


# %%
# functions
def parse_ann_file(ann_path):
    """
    Parses .ann file for entities (T), binary relations (R) and n-ary relation (*).
    Returns:
        entities: dict {id: {type, start, end, text}}
        relations: list of tuples (type, arg1_id, arg2_id)
    """
    
    entities = dict()
    relations = list()
    with open(ann_path, "r", encoding="UTF-8") as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            # parse entities (T)
            if line.startswith("T"):
                parts = line.split("\t")
                
                if len(parts) < 3:
                    continue
                
                entity_id = parts[0]
                # discontinuous spans -> "1311 1318;1334 1341"
                # -> take min start and max end 
                entity_offsets = parts[1].split(' ')
                entity_type = entity_offsets[0]
                
                # extract offset indices as integers
                indices = [int(x) for x in re.findall(r"\d+", parts[1])]
                if not indices:
                    continue
                start = min(indices)
                end = max(indices)
                
                entities[entity_id] = {
                    "id": entity_id,
                    "type": entity_type,
                    "start": start,
                    "end": end,
                    "text": parts[2]
                }
                
            # parse directed binary relations (R)
            # format: R1	Has_temporal Arg1:T6 Arg2:T7
            elif line.startswith("R"):
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                
                args_part = parts[1].split(" ")
                relation_type = args_part[0]

                # extract relation arguments (entities)
                arg1 = None
                arg2 = None
                for arg in args_part:
                    if arg.startswith("Arg1:"):
                        arg1 = arg.split(":")[1]
                    elif arg.startswith("Arg2:"):
                        arg2 = arg.split(":")[1]
                
                # build list of relation-tuples
                if arg1 and arg2:
                    relations.append((relation_type, arg1, arg2))
            
            # parse OR-relations
            elif line.startswith("*"):
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                
                args_part = parts[1].split(" ")
                relation_type = args_part[0]
                args = args_part[1:]
                
                # decompose n-ary set into pairwise permutations
                # generate (arg1, arg2) and (arg2, arg1) for all arguments
                if len(args) >= 2:
                    for arg1, arg2 in itertools.permutations(args, 2):
                        relations.append((relation_type, arg1, arg2))


    return entities, relations



def process_files(data_dir):
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    samples = list()
    
    # set to collect all relations that appear in data
    all_relation_types = set()
    
    print(f"Parsing {len(txt_files)} files for relation extraction...")
    
    for txt_path in txt_files:
        ann_path = txt_path.replace(".txt", ".ann")
        if not os.path.exists(ann_path):
            continue
        
        # extract criteria type from filename
        file_name = os.path.basename(txt_path)
        if "_exc" in file_name:
            criteria_type = "exclusion"
        elif "_inc" in file_name:
            criteria_type = "inclusion"
        else:
            criteria_type = "unknown"
        
        with open(txt_path, "r", encoding="UTF-8") as f:
            text = f.read()
        
        entities, relations = parse_ann_file(ann_path)
        
        # update set of relations
        for rel in relations:
            all_relation_types.add(rel[0])
        
        # mapping dict
        # (arg1, arg2) -> relation_type
        # ACHTUNG, GEHT DAVON AUS, DASS GLEICHE ENTITIES 
        # IMMER IN GLEICHER RELATION STEHEN!
        relation_lookup = {(r[1], r[2]): r[0] for r in relations}
        
        # split into sentences (newline split to match NER logic)
        # ACHTUNG: coss-sentence relations are skipped 
        lines = text.split('\n')
        global_offset = 0
        
        for line in lines:
            line_len = len(line)
            line_end = global_offset + line_len
            
            # entities in line
            line_entities = list()
            for entity_id, entity in entities.items():
                if entity["start"] >= global_offset and entity["end"] <= line_end:
                    line_entities.append(entity)
            
            # relation generation with entity pairs
            # less than two entities -> no relation
            if len(line_entities) >= 2:
                # generating all permutations (arg1, arg2)
                # permutations because relations are directional (arg1->arg2 != arg2->arg1)
                for e1, e2 in itertools.permutations(line_entities, 2):
                    
                    # ground truth from mapping dict
                    label = relation_lookup.get((e1["id"], e2["id"]), "NO_RELATION")
                    
                    # local indices
                    # for [E1] markers injection later
                    e1_start_local = e1["start"]-global_offset
                    e1_end_local = e1["end"]-global_offset
                    e2_start_local = e2["start"]-global_offset
                    e2_end_local = e2["end"]-global_offset
                    
                    samples.append({
                        "text": line,
                        "e1_start": e1_start_local,
                        "e1_end": e1_end_local,
                        "e1_type": e1["type"],
                        "e2_start": e2_start_local,
                        "e2_end": e2_end_local,
                        "e2_type": e2["type"],
                        "label": label,
                        "criteria_type": criteria_type,
                        "filename": file_name
                    })
            
            # +1 for newline
            global_offset += line_len + 1 
    
    relations_df = pd.DataFrame(samples)
    return relations_df, all_relation_types



def split_and_save_dataset(dataset, output_dir, seed=42):
    """
    Splits the generated training dataset into train, eval and test.
    Applies stratification on the 'label' column to handle class imbalance.
    Simple stratification is enough, because every datapoint only has one label.
    """
    
    print("Splitting dataset into train/eval/test...")
    
    # first split
    split_1 = dataset.train_test_split(test_size=0.1, stratify_by_column="label", seed=seed)
    test_dataset = split_1["test"]
    remaining_dataset = split_1["train"]
    
    # second split
    # 0.1111 * 0.9 -> 0.10
    split_2 = remaining_dataset.train_test_split(test_size=0.1111, stratify_by_column="label", seed=seed)
    train_dataset = split_2["train"]
    eval_dataset = split_2["test"]
    
    # combine splits into on dataset
    final_dataset = DatasetDict({
        "train": train_dataset,
        "validation": eval_dataset,
        "test": test_dataset
    })
    
    # save dataset to disk
    print(f"Saving dataset to {output_dir}...")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(eval_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    
    final_dataset.save_to_disk(output_dir)
    print("Save complete.")
    
    return None




# %%
# Configurations
DATA_DIR = "./data/chia_without_scope_test"
OUTPUT_DIR = "./data/chia_without_scope_parsedRE_TEST"

# %%
# boilerplate
if __name__ == "__main__":
    # load and parse chia-files
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found")
    else:
        df, relation_types_found = process_files(DATA_DIR)
        print(f"Generated {len(df)} relation pairs")
        print(df["label"].value_counts())
    
    # %%
    # dynamic set relation types
    # NO_RELATION = 0
    relation_types = ["NO_RELATION"]
    relation_types.extend(sorted(list(relation_types_found)))
    print(f"Relation types found: {len(relation_types)} - {relation_types}")
    
    
    # %%
    # create HuggingFace dataset
    # and cast label to ClassLabel
    # NO_RELATION = 0
    dataset = Dataset.from_pandas(df)
    class_label = ClassLabel(names=relation_types)
    dataset = dataset.cast_column("label", class_label)
    
    # %%
    # split and save dataset
    # generate and save label maps
    split_and_save_dataset(dataset, OUTPUT_DIR)
    
    label2id = {l: i for i, l in enumerate(relation_types)}
    id2label = {i: l for l, i in label2id.items()}
    map_path = os.path.join(OUTPUT_DIR, "label_map.json")
    with open(map_path, "w", encoding="UTF-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=4)
        print(f"Label map saved to {map_path}")