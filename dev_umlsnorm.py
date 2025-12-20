# %%
# imports
import os
import numpy as np
import pandas as pd
import faiss
import torch
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel



# %%
# model (SapBERT = UMLS-trained BERT)
model_checkpoint = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)


# %%
# text encoding
def encode_text(text, batch_size=32):
    
    embeddings = list()
    
    for idx in tqdm(range(0, len(text), batch_size)):
        batch = text[idx:idx + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        cls_embeddings = outputs.last_hidden_state[:, 0]
        embeddings.append(cls_embeddings.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return embeddings


# %%
# load umls synonyms
# NEED LOCAL UMLS DICTIONARIES!
def load_umls_synonyms(
    mrconso_path,
    allowed_sources=None,
    allowed_semantic_types=None,
    mrsty_path=None
    ):
    """
    Returns:
      texts: list[str]
      cui_list: list[str]
    """

    cols = [
        "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF",
        "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS"
    ]

    df = pd.read_csv(
        mrconso_path,
        sep="|",
        header=None,
        names=cols,
        usecols=["CUI", "LAT", "STR", "SAB"],
        dtype=str
    )

    # English only
    df = df[df["LAT"] == "ENG"]

    # Optional source filtering
    if allowed_sources is not None:
        df = df[df["SAB"].isin(allowed_sources)]

    # Optional semantic type filtering
    if allowed_semantic_types is not None and mrsty_path is not None:
        sty = pd.read_csv(
            mrsty_path,
            sep="|",
            header=None,
            names=["CUI", "TUI", "STN", "STY", "ATUI", "CVF"],
            usecols=["CUI", "TUI"],
            dtype=str
        )
        sty = sty[sty["TUI"].isin(allowed_semantic_types)]
        df = df[df["CUI"].isin(sty["CUI"])]

    texts = df["STR"].str.lower().tolist()
    cui_list = df["CUI"].tolist()

    return texts, cui_list

# %%
# mirror what will be mrconso + mrsty
MOCK_UMLS = {
    "C0020538": {
        "preferred_name": "Hypertension",
        "semantic_type": "Disease",
        "synonyms": [
            "hypertension",
            "high blood pressure",
            "arterial hypertension"
        ],
        "icd10": ["I10"]
    },
    "C0011849": {
        "preferred_name": "Diabetes Mellitus, Type 2",
        "semantic_type": "Disease",
        "synonyms": [
            "type 2 diabetes",
            "diabetes mellitus type 2",
            "adult onset diabetes"
        ],
        "icd10": ["E11"]
    },
    "C0018799": {
        "preferred_name": "Heart Attack",
        "semantic_type": "Disease",
        "synonyms": [
            "heart attack",
            "myocardial infarction",
            "acute myocardial infarction"
        ],
        "icd10": ["TEST1"]
    },
    "C0004057": {
        "preferred_name": "Aspirin",
        "semantic_type": "Drug",
        "synonyms": [
            "aspirin",
            "acetylsalicylic acid",
            "asa"
        ],
        "icd10": ["TEST2"]
    }
}

# development purposes
def mock_umls_to_index(mock_umls):
    texts = []
    cuis = []
    meta = []

    for cui, entry in mock_umls.items():
        for synonym in entry["synonyms"]:
            texts.append(synonym.lower())
            cuis.append(cui)
            meta.append({
                "cui": cui,
                "preferred_name": entry["preferred_name"],
                "semantic_type": entry["semantic_type"],
                "synonym": synonym
            })

    return texts, cuis, meta



# %%
# build faiss index#
# faiss-library for fast similarity search
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# %%
# entity linking
def link_entity(
    mention,
    index,
    umls_texts,
    umls_cuis,
    top_k=10
):
    mention = mention.lower().strip()

    query_vec = encode_text([mention])
    scores, indices = index.search(query_vec, top_k)

    cui_scores = defaultdict(float)

    for score, idx in zip(scores[0], indices[0]):
        cui = umls_cuis[idx]
        cui_scores[cui] = max(cui_scores[cui], float(score))

    best_cui = max(cui_scores, key=cui_scores.get)

    return {
        "mention": mention,
        "cui": best_cui,
        "score": cui_scores[best_cui]
    }



# %%
# main pipeline example
# boilerplate
if __name__ == "__main__":

    # -------- CONFIG --------
    UMLS_DIR = "/path/to/umls"
    MRCONSO = os.path.join(UMLS_DIR, "MRCONSO.RRF")
    MRSTY = os.path.join(UMLS_DIR, "MRSTY.RRF")

    # Restrict to diseases (recommended!)
    DISEASE_TUIS = {
        "T047",  # Disease or Syndrome
        "T048",  # Mental or Behavioral Dysfunction
        "T191"   # Neoplastic Process
    }

    SOURCES = {
        "SNOMEDCT_US",
        "ICD10CM",
        "MSH"
    }

    # -------- LOAD UMLS --------
    print("Loading UMLS synonyms...")
    umls_texts, umls_cuis = load_umls_synonyms(
        MRCONSO,
        allowed_sources=SOURCES,
        allowed_semantic_types=DISEASE_TUIS,
        mrsty_path=MRSTY
    )
    
    # %%
    # (1)
    # -------- MOCKUP - LOAD UMLS --------
    umls_texts, umls_cuis, umls_meta = mock_umls_to_index(MOCK_UMLS)
    print(f"Loaded {len(umls_texts)} UMLS strings")

    # %%
    # (2)
    # -------- ENCODE & INDEX --------
    print("Encoding UMLS strings...")
    umls_embeddings = encode_text(umls_texts)

    # %%
    # (3)
    print("Building FAISS index...")
    index = build_faiss_index(umls_embeddings)

    
    # %%
    # (4)
    # -------- LINK EXAMPLE MENTIONS --------
    mentions = [
        "uncontrolled hypertension",
        "type 2 diabetes",
        "heart attack",
        "auto-immune disease , acute stage"
    ]

    for mention in mentions:
        result = link_entity(
            mention,
            index,
            umls_texts,
            umls_cuis
        )
        print(result)
        cui = result["cui"]
        icd_codes = MOCK_UMLS[cui].get("icd10", [])
        print(f"ICD CODE: {icd_codes}")
