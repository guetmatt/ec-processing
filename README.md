DATA PATHS AND ARGUMENT VALUES TO BE ADJUSTED!


# Clinical Trial Information Extraction - NER + RE
This repository contains the implementation of a modular NLP pipeline for processing clinical trial eligibility criteria. It leverages domain-specific BERT models (e.g., Bio_ClinicalBERT) to identify medical entities and the semantic relations between them.

## Project Structure
src/
├── ner_parseChia.py      # Preprocesses Chia dataset for NER tasks
├── ner_training.py       # Training and HPO for the NER model
├── re_parseChia.py       # Preprocesses Chia dataset for RE tasks
├── re_training.py        # Training and HPO for the RE model
├── parse_ctg.py          # Parses raw data from ClinicalTrials.gov (CTG)
└── pipeline_inference.py # End-to-end inference (NER -> Pair Generation -> RE)

## (1) Data Preprocessing
Before training, the raw chia_without_scope dataset (in BRAT format) must be converted into a format compatible with HuggingFace datasets.

### Named Entity Recognition (NER)
Parses ```.ann``` and ```.txt``` files into tokenized BIO-labeled sequences.
```python ner_parseChia.py --data_dir ../data/chia_raw --output_dir ../data/processed_ner```

### Relation Extraction (RE)
Generates entity pairs within sentences. Includes negative sampling (NO_RELATION) and negative downsampling to manage class imbalance.
```python re_parseChia.py --data_dir ../data/chia_raw --output_dir ../data/processed_re --train_downsample_rate 0.2```

## (2) Model Training
Both modules support Hyperparameter Optimization (HPO) via Optuna and are configured to use Bio_ClinicalBERT by default.

### Training the NER Module
Trains a token-classification model. Supports both span-based (seq) and token-based (tok) evaluation.
```python ner_training.py --data_dir ../data/processed_ner --output_dir ../models/ner_model --eval_method seq --hpo_trials 10```

### Training the RE Module
Trains a sequence-classification model. This script automatically injects entity markers (e.g., ```[E1]...[/E1]```) to provide the model with positional context of the arguments.
```python re_training.py --data_dir ../data/processed_re --output_dir ../models/re_model --do_hpo --hpo_trials 10```

## (3) ClinicalTrials.gov (CTG) Data Processing
To apply the models to real-world data, use the CTG parser to convert CSV exports into sentence-level criteria.
```python parse_ctg.py --input_path ../data_ctg/raw_trials.csv --output_path ../data_ctg/parsed_sentences.csv```

## (4) End-to-End Inference Pipeline
The ```pipeline_inference.py``` script connects both modules. It performs the following workflow:
- 1. NER: Detects entities in raw text.
- 2. Candidate Pairing: Generates all valid permutations of detected entities.
- 3. RE: Classifies the relationship for each pair.

```python pipeline_inference.py \```
    ```--data_dir ./data_ctg/parsed_sentences.csv \```
    ```--ner_model_path ./models/ner_model \```
    ```--re_model_path ./models/re_model \```
    ```--output_dir ./results/v1 \```
    ```--mode ner+re``` 

## Key Technical Features
- Iterative Stratification: Used in ```ner_parseChia.py``` to ensure multi-label balance across train/val/test splits.
- Entity Marker Injection: The RE module uses a right-to-left injection method to prevent character index shifting during preprocessing.
- Weighted Loss: ```re_training.py``` implements a square-root dampened class weighting to handle the high frequency of "NO_RELATION" samples.
- Evaluation: Detailed reporting includes seqeval for NER spans and normalized confusion matrices for RE.

## Requirements
- Python 3.8+
- PyTorch
- Transformers & Datasets (HuggingFace)
- Optuna (for HPO)
- Evaluate & Seqeval
- Scikit-learn, Pandas, Numpy