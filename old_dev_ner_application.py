# # apply a trained NER-model to the ctg-dataset

# # %%
# # imports
# import pandas as pd
# import json
# from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# # %%
# # functions
# def load_ner_pipeline(model_checkpoint, device=-1):
#     """
#     Initializes the HuggingFace token classification pipeline.
#     The 'simple' aggregation strategy is crucial for BERT-based NER as it automatically
#     merges 'B-' (Beginning) and 'I-' (Inside) sub-tokens into coherent word-level entities,
#     simplifying downstream processing.

#     Args:
#         model_checkpoint (str): file path to the trained model, used as tokenizer and NER-classifier
#         device (int): device index to run the model on (-1 for CPU, 0+ for GPU)

#     Returns:
#         pipeline: a configured Hugging Face pipeline token classification.
#     """
    
#     ner_pipeline = pipeline(
#         "token-classification",
#         model=model_checkpoint,
#         tokenizer=model_checkpoint,
#         aggregation_strategy="simple",
#         device=device
#         )
    
#     return ner_pipeline



# def extract_entities(text, ner_pipeline):
#     """
#     Extracts named entities from a single sentence using the pre-loaded NER pipeline.

#     This function processes the input text and formats the pipeline's raw output 
#     into a clean, JSON-serializable list of dictionaries.

#     Args:
#         text (str): input sentence for NER
#         ner_pipeline (Any): loaded NER pipeline (from load_ner_pipeline)

#     Returns:
#         List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents 
#         a detected entity with the following keys:
#             - 'entity_group': The predicted label (e.g., 'Drug', 'Disease').
#             - 'word': The text span of the entity.
#             - 'start': The start character index.
#             - 'end': The end character index.
#             - 'score': The confidence score of the prediction.
#     """
#     if not isinstance(text, str) or not text.strip():
#         return []
    
#     # run inference
#     predictions = ner_pipeline(text)
    
#     # format results for JSON serialization
#     cleaned_entities = list() 
#     for pred in predictions:
#         clean_pred = {
#             "entity_group": pred["entity_group"],
#             "word": pred["word"],
#             "start": int(pred["start"]),
#             "end": int(pred["end"]),
#             "score": float(pred["score"])
#         }
#         cleaned_entities.append(clean_pred)

    
#     return cleaned_entities



# def process_dataset_ner(input_file: str, output_file: str, model_path: str):
#     """
#     NER application workflow on the full ctg-dataset.
#     Loads the processed ctg-dataset, applies the NER model to every sentence,
#     and saves the results with a new 'Entities' column containing JSON data.

#     Args:
#         input_file (str): path to the processed ctg-dataset (csv file)
#         output_file (str): path where the result csv will be saved
#         model_checkpoint (str): path to the trained NER model
#     """
    
#     print(f"Loading data from {input_file}...")
#     df = pd.read_csv(input_file)
    
#     print(f"Loading model from {model_path}...")
#     ner_pipe = load_ner_pipeline(model_path)
    
#     print("Running NER inference on all sentences...")
#     entities_column = [
#         json.dumps(extract_entities(text, ner_pipe)) 
#         for text in df['Sentence']
#     ]
    
#     # Assign result and save
#     df['Entities'] = entities_column
#     df.to_csv(output_file, index=False)
#     print(f"Inference complete. Results saved to {output_file}")











# # %%
# # boilerplate
# if __name__ == "__main__":
    
#     DATA_DIR = "./data_by_lea/data_parsed_sentLevel/ctg_sentLevel.csv"
#     # OUTPUT_FILE = "./results/ner_dev/dev_ner_results.csv"
#     OUTPUT_FILE = "./results/ner_dev/dev_ner_results.jsonl"
#     MODEL_CHECKPOINT = "./models/ner_chia_test2"
    

#     # %%
#     # load dataset  
#     print(f"Loading data from {DATA_DIR}...")
#     df = pd.read_csv(DATA_DIR)
#     # test fragment of dataset
#     df = df[:100]
    
    
#     # %%
#     # load NER pipeline
#     print(f"Loading model from {MODEL_CHECKPOINT}...")
#     ner_pipeline = load_ner_pipeline(MODEL_CHECKPOINT)
    
    
#     # %%
#     # run NER pipeline on dataset
#     print("Running NER inference on all sentences...")
    
#     # JSONL-file generation
#     results = list()
#     for idx, row in df.iterrows():
#         text = row["Sentence"]
#         entities = extract_entities(text, ner_pipeline)
#         # dict for jsonl-storage
#         entry = {
#             "sentence_id": idx,
#             "text": text,
#             "entities": entities,
#             "nct_id": row["StudyNCTid"],
#             "criteria_type": row["criteria_type"]
#         }
#         results.append(entry)
    
#     # %%
#     # save jsonl-format
#     with open(OUTPUT_FILE, "w", encoding="UTF-8") as f:
#         for entry in results:
#             json.dump(entry, f)
#             f.write("\n")
#     print(f"Results saved to {OUTPUT_FILE}")
    
    
    
#     # # CSV-FILE GENERATION
#     # entities_column = [
#     #     json.dumps(extract_entities(text, ner_pipeline)) 
#     #     for text in df['Sentence']
#     # ]
    
#     # # %%
#     # # assign result and save
#     # df['Entities'] = entities_column
#     # df.to_csv(OUTPUT_FILE, index=False)
#     # print(f"Inference complete. Results saved to {OUTPUT_FILE}")