# parsing data from csv-file from ClinicalTrials.gov

# %%
# imports
import pandas as pd
import re


# %%
# functions
def parse_criteria_text(text):
    """
    Parses a single block of clinical trial eligibility criteria
    text into structured sentence-level data.
    
    Args:
        text (str): The raw eligibility criteria text
    
    Returns:
    """
    
    lines = text.split("\n")
    parsed_items = list()
    current_section = None
    
    # regex patterns
    inclusion_pattern = re.compile(r"^\s*inclusion\s+criteria:?", re.IGNORECASE)
    exclusion_pattern = re.compile(r'^\s*exclusion\s+criteria:?', re.IGNORECASE)
    bullet_pattern = re.compile(r'^\s*[*-]\s+(.*)')

    # process lines
    for line in lines:
        line_stripped = line.strip()
        
        # ignore empty lines
        if not line_stripped:
            continue
        
        # header lines
        if inclusion_pattern.search(line_stripped):
            current_section = "inclusion"
            continue
        elif exclusion_pattern.search(line_stripped):
            current_section = "exclusion"
            continue
        
        # content criteria lines
        if current_section:
            match = bullet_pattern.match(line_stripped)
            if match:
                content = match.group(1).strip()
                if content:
                    parsed_items.append({
                        "criteria_type": current_section,
                        "text": content
                    })
    
    return parsed_items



def process_dataset(input_path, output_path):
    """
    Loads, parsed and saves the ctg-dataset.
    """
    
    # load dataset
    df = pd.read_csv(input_path)
    
    # parse dataset
    processed_data = list()
    for _, row in df.iterrows():
        for item in parse_criteria_text(row["EligibilityCriteria"]):
            processed_row = {
                "StudyNCTid": row["StudyNCTid"],
                "criteria_type": item["criteria_type"],
                "Sentence": item["text"]
                }
            processed_data.append(processed_row)
    
    # save processed dataset as csv
    df_processed = pd.DataFrame(processed_data)
    df_processed.to_csv(output_path, index=False)
    
    return df_processed


# %%
# boilerplate
if __name__ == "__main__":
    
    DATA_DIR = "./data_by_lea/Studies_with_id_and_EligibilityCriteria.csv"
    OUTPUT_DIR = "./data_by_lea/data_parsed_sentLevel/ctg_sentLevel.csv"
    
    process_dataset(DATA_DIR, OUTPUT_DIR)