# (1) entity marker insertion -- DONE
# (2) pytorch dataset -- DONE?
# (3) trainer pipeline


# DOES THIS CODE ALSO WORK WITH WHOLE DATASET DIRECTORY?

# LOAD AND PREPROCESS CHIA DATASET 

# %%
# imports
from pathlib import Path
from itertools import permutations
from datasets import Dataset
from datasets import load_from_disk, DatasetDict
import os



# %%
# read files
def load_text(filepath: str):
    with open(filepath, "r", encoding="UTF-8") as f:
        return f.read()
    
def load_lines(filepath: str):
    with open(filepath, "r", encoding="UTF-8") as f:
        return [line.rstrip("\n") for line in f]



# %%
# automatically extract criterion type
def infer_criterion_type(filepath):
    name = filepath.stem.lower()
    
    if "_inc" in name:
        return "inclusion"
    elif "_exc" in name:
        return "exclusion"
    else:
        raise ValueError(f"Cannot infer criterion type from filename: {filepath}")



# %%
# parse .ann file
def parse_ann(path):
    entities = {}
    relations = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # --------
            # Entities
            # --------
            if line.startswith("T"):
                tid, span_info, text = line.split("\t")
                parts = span_info.split()
                label = parts[0]
                span_str = " ". join(parts[1:])
                starts = []
                ends = []

                if ";" in span_str:
                    span_parts = span_str.split()
                    s = span_parts[0]
                    e = span_parts[-1]
                    starts.append(int(s))
                    ends.append(int(e))
                # continuous entity spans
                else:
                    s, e = span_str.split()
                    starts.append(int(s))
                    ends.append(int(e))

                entities[tid] = {
                    "id": tid,
                    "type": label,
                    "start": min(starts),
                    "end": max(ends),
                    "text": text
                }

            # ---------
            # Binary Relations - R
            # ---------
            elif line.startswith("R"):
                parts = line.split()
                rid = parts[0]
                label = parts[1]
                arg1 = parts[2].split(":")[1]
                arg2 = parts[3].split(":")[1]

                relations.append({
                    "id": rid,
                    "label": label,
                    "head": arg1,
                    "tail": arg2
                })
                
               
            # -------------------
            # N-ary logical relations - *
            # -------------------
            elif line.startswith("*"):
                parts = line.split()
                label = parts[1]          # e.g. OR, AND, NOT
                args = parts[2:]          # [T1, T2, T3, ...]

                # convert to pairwise relations
                for i in range(len(args)):
                    for j in range(i + 1, len(args)):
                        relations.append({
                            "id": f"*{label}_{args[i]}_{args[j]}",
                            "label": label,
                            "head": args[i],
                            "tail": args[j]
                        }) 

    return entities, relations



# %%
# split text into criteria with character offsets
def split_criteria(text):
    """
    Returns:
      - list of (criterion_text, start_offset, end_offset)
    """
    criteria = []
    offset = 0

    for line in text.splitlines(keepends=True):
        start = offset
        end = start + len(line.rstrip("\n"))
        criteria.append((line.rstrip("\n"), start, end))
        offset += len(line)

    return criteria



# %%
# assign entities & relations to criteria
# CRITERION TYPE SHOULD BE AUTOMATICALLY EXTRACTED
def build_dataset(txt_path, ann_path):
    raw_text = load_text(txt_path)
    criteria = split_criteria(raw_text)

    entities, relations = parse_ann(ann_path)
    criterion_type = infer_criterion_type(txt_path)

    dataset = []

    for idx, (crit_text, start, end) in enumerate(criteria):
        
        crit_entities = []
        ent_id_map = {}
        for eid, ent in entities.items():
            if start <= ent["start"] < end:
                local_ent = ent.copy()
                local_ent["start"] -= start
                local_ent["end"] -= start
                crit_entities.append(local_ent)
                ent_id_map[eid] = local_ent["id"]

        crit_relations = []
        for rel in relations:
            if rel["head"] in ent_id_map and rel["tail"] in ent_id_map:
                crit_relations.append({
                    "head": rel["head"],
                    "tail": rel["tail"],
                    "label": rel["label"]
                })

        dataset.append({
            "criterion_id": f"{txt_path.stem}_{idx}",
            "trial_id": txt_path.stem.split("_")[0],
            "text": crit_text,
            "entities": crit_entities,
            "relations": crit_relations,
            "criterion_type": criterion_type
        })

    return dataset


# %%
# build dataset from multiple files
def build_dataset_from_dir(data_dir: Path):
    """
    Build a Chia dataset from all .txt/.ann file pairs in a directory.

    Args:
        data_dir (Path): directory containing Chia files

    Returns:
        dataset (list): unified dataset (criterion-level)
    """
    dataset = []

    txt_files = sorted(data_dir.glob("*.txt"))

    for txt_path in txt_files:
        ann_path = txt_path.with_suffix(".ann")

        if not ann_path.exists():
            raise FileNotFoundError(
                f"Missing annotation file for {txt_path.name}"
            )

        # build dataset for this file
        file_dataset = build_dataset(
            txt_path=txt_path,
            ann_path=ann_path
        )

        dataset.extend(file_dataset)

    return dataset




# %%
# example run
data_dir = Path("./data/chia_without_scope")
txt_file = Path("./data/chia_without_scope/NCT00050349_exc.txt")
ann_file = Path("./data/chia_without_scope/NCT00050349_exc.ann")


# dataset = build_dataset_from_dir(data_dir)

# %%
# print(f"TOTAL CRITERIA: {len(dataset)}")
dataset = build_dataset(
    txt_path=txt_file,
    ann_path=ann_file)

# %%
print(dataset[0])
      



# %%
# entity marker insertion
# COMMENTED FROM GEMINI
# has to be done AFTER building pairwise dataset
# or within the function of building pairwise dataset 
def insert_entity_markers(text, e1, e2):
    """
    Injects [E1], [/E1], [E2], [/E2] markers into the text string.
    """
    # Define insertions: (position, tag, priority)
    # Priority 2 (Start) > Priority 1 (End) helps when e1 and e2 start at the same position.
    insertions = [
        (e1["start"], "[E1]", 2),
        (e1["end"], "[/E1]", 1),
        (e2["start"], "[E2]", 2),
        (e2["end"], "[/E2]", 1)
    ]
    
    # Sort by position (descending), then by priority (ascending) 
    # This ensures that if we have "word", we get "[E1]word[/E1]" 
    # and not markers mixed up or inserted in wrong order.
    insertions.sort(key=lambda x: (x[0], x[2]), reverse=True)
    
    marked_text = text
    for pos, marker, _ in insertions:
        # Check bounds to avoid errors if spans are messy
        pos = max(0, min(len(marked_text), pos))
        marked_text = marked_text[:pos] + marker + marked_text[pos:]
        
    return marked_text


# %%
# build pairwise RE dataset
def build_pairwise_re_dataset(
    dataset,
    label_map=None,
    include_self_relations=False,
    no_relation_label="NO_RELATION"
):
    """
    Convert a criterion-level Chia dataset into pairwise RE samples.

    Args:
        dataset (list): Output of build_dataset()
        label_map (dict): Optional mapping from raw labels to normalized labels
        include_self_relations (bool): Whether to include (e, e) pairs
        no_relation_label (str): Label for entity pairs without relations

    Returns:
        pairwise_samples (list): Pairwise RE dataset
    """

    pairwise_samples = []

    for item in dataset:
        text = item["text"]
        entities = item["entities"]
        relations = item.get("relations", [])
        criterion_id = item["criterion_id"]
        criterion_type = item.get("criterion_type")

        # index relations for fast lookup
        rel_lookup = {}
        for r in relations:
            label = r["label"]
            if label_map:
                label = label_map.get(label, label)
            rel_lookup[(r["head"], r["tail"])] = label

        # generate entity pairs
        for e1, e2 in permutations(entities, 2):
            if not include_self_relations and e1["id"] == e2["id"]:
                continue

            label = rel_lookup.get(
                (e1["id"], e2["id"]),
                no_relation_label
            )
            
            # entity marker insertion
            marked_text = insert_entity_markers(text, e1, e2)

            pairwise_samples.append({
                "criterion_id": criterion_id,
                "text": text,
                "marked_text": marked_text,
                "e1": {
                    "id": e1["id"],
                    "type": e1["type"],
                    "start": e1["start"],
                    "end": e1["end"],
                    "text": e1["text"]
                },
                "e2": {
                    "id": e2["id"],
                    "type": e2["type"],
                    "start": e2["start"],
                    "end": e2["end"],
                    "text": e2["text"]
                },
                "label": label,
                "criterion_type": criterion_type
            })

    return pairwise_samples


# %%
pairwise_relations = build_pairwise_re_dataset(dataset)

# %%
for rel in pairwise_relations:
    print(rel)

# %%    
print(len(pairwise_relations))



# %%
# save preprocessed dataset to disk
# MAYBE SPLIT INTO TRAIN, TEST, VAL WHEN SAVING??
def save_dataset(data_list, save_dir):
    """
    Saves the entire dataset to disk without splitting.
    """
    
    # convert list of dicts to Huggingface Dataset
    full_dataset = Dataset.from_list(data_list)
    
    # Save to disk
    os.makedirs(save_dir, exist_ok=True)
    full_dataset.save_to_disk(save_dir)
    
    print(f"Dataset saved to '{save_dir}'")
    
    return None



# %%
# load huggingface dataset    
def load_dataset(load_dir: str):
    """
    Loads a Huggingface Dataset from disk.
    
    Args:
    - load_dir (str): Path to the saved dataset directory
    
    Returns:
    - Dataset
    """
    
    try:
        dataset = load_from_disk(load_dir)
        print(f"Loaded datset from '{load_dir}' with {len(dataset)} samples.")
        return dataset
    except FileNotFoundError:
        print(f"Error: No dataset found at '{load_dir}'")
        return None
    

# %%
# split huggingface dataset to train/test/val
def split_dataset(dataset,
                  test_size=0.1,
                  val_size=0.1,
                  seed=42):
    """
    Splits a Huggingface Dataset into train, validation, test splits.
    
    Args:
        full_dataset (Dataset): The dataset object to split.
        test_size (float): Absolute fraction for Test set (e.g., 0.1 for 10%).
        val_size (float): Absolute fraction for Validation set (e.g., 0.1 for 10%).
        seed (int): Random seed for reproducibility.
        
    Returns:
        DatasetDict: A dictionary containing 'train', 'validation', and 'test'.
    """
    
    # calculate size of temporary holdout set (test+val)
    holdout_total_size = test_size + val_size
    
    # split train from holdout
    try:
        split_1 = dataset.train_test_split(
            test_size=holdout_total_size,
            seed=seed,
            stratify_by_column="label"
        )
    except ValueError:
        print("Warning: Stratification failed (classes likely too small). Splitting randomly.")
        split_1 = dataset.train_test_split(
            test_size=holdout_total_size,
            seed=seed
        )
        
    ds_train = split_1["train"]
    ds_holdout = split_1["test"]
    
    # split val from test
    relative_test_size = test_size/holdout_total_size
    
    split_2 = ds_holdout.train_test_split(
        test_size=relative_test_size,
        seed=seed
    )
    
    ds_val = split_2["train"]
    ds_test = split_2["test"]
    
    # return organized dataset
    ds_new = DatasetDict({
        "train": ds_train,
        "test": ds_test,
        "validation": ds_val
    })
    
    return ds_new



# %%
SAVE_PATH = "./data/chia_parsed_v1"
#%%
save_dataset(pairwise_relations, SAVE_PATH)



# %%
dataset = load_dataset("./data/chia_parsed_v1")

# %%
ds_split = split_dataset(
    dataset,
    test_size=0.1,
    val_size=0.1,
    seed=42
)

# %%
set(ds_split["train"]["label"])

#%%
print("\nSplit Sizes:")
print(f"Train: {len(ds_split['train'])}")
print(f"Val:   {len(ds_split['validation'])}")
print(f"Test:  {len(ds_split['test'])}")