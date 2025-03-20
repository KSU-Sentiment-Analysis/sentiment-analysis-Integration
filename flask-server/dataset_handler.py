import pandas as pd
import os
from fuzzywuzzy import process
import re
import numpy as np

# Using fuzzywuzzy for column matching
def load_csv(dataset_name="dataset_1.csv"):
    """
    Loads a CSV file dynamically with improved error handling.
    """
    file_path = os.path.join(os.getcwd(), dataset_name)

    if not os.path.exists(file_path):
        print(f"Error: {dataset_name} not found in the current directory")
        return None

    # Try different encoding types
    encodings = ["utf-8", "ISO-8859-1"]
    delimiters = [",", ";", "\t"]  #  delimiters add more if u find more or other cases

    for encoding in encodings:
        for delimiter in delimiters:
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    sep=delimiter,
                    on_bad_lines="skip",  # Skipping broken rows
                    engine="python"
                )
                print(f"Loaded {dataset_name} with encoding={encoding} and delimiter='{delimiter}'")
                print(f"Columns detected: {df.columns.tolist()}\n")
                return df
            except Exception as e:
                print(f"⚠Failed with encoding={encoding}, delimiter='{delimiter}': {e}")

    print(f"Could not load {dataset_name}. file corrupt???")
    return None


# Here where we define common col names for reviews and review text stuff
EXPECTED_COLUMNS = {
    "review_text": ["review text", "text", "comment", "feedback", "opinion", "customer review"],
    "rating": ["rating", "score", "stars", "rank"]
}


def fuzzy_match_column(df, column_type):
    """
    Finds the best-matching column name in the dataset using fuzzy matching,
    ensuring proper filtering of unwanted columns and prioritizing known correct names.
    :param df: DataFrame containing dataset columns
    :param column_type: Type of column to search for ('review_text' or 'rating')
    :return: Best matched column name or None if no match found
    """
    if column_type not in EXPECTED_COLUMNS:
        return None

    # before we use fuzzy wusy just gonna do a manual check just in case we can save reousrcez.
    #for col in df.columns:
    #    col_clean = col.lower().replace(" ", "_")
    #    if column_type == "review_text" and col_clean == "review_text":
    #        print(f"Manually select '{col}' as the review text column.")
    #        return col
    #    if column_type == "rating" and col_clean == "rating":
    #        print(f"Manually select '{col}' as the rating column.")
    #        return col

    # now we use fuzzy wuszzy
    best_match = None
    highest_score = 0

    for col in df.columns:
        col_clean = col.lower().replace(" ", "_")

        # Ignore name columns and such
        if "name" in col_clean or "count" in col_clean or "profile" in col_clean:
            continue

        # here is where we extractr best matching words fuzzywuzzy
        match_result = process.extractOne(col_clean, EXPECTED_COLUMNS[column_type])
        if match_result:
            match, score = match_result
            if score > highest_score and score >= 80:  # 80% confidence threshold might need to expierment but it works so far
                best_match = col
                highest_score = score

    return best_match


def extract_numeric_rating(rating_text):
    """
    Extracts a numerical rating from a text-based rating string. basically a whole bunch of batter detection

    Works for:
      - "Rated 5 out of 5 stars" → 5
      - "Score: 3.5" → 3.5
    :param rating_text: Raw text from the rating column
    :return: Extracted numerical rating (float) or NaN if extraction fails
    """
    rating_text = str(rating_text).lower().strip()
    match = re.search(r"(\d+)\s*(?:out\s*of|\/)\s*\d+", rating_text)
    if match:
        return int(match.group(1))

    try:
        return float(rating_text)
    except ValueError:
        return np.nan


def process_data(df):
    """
    Here is where i process the fuzzy matches words to standerize so our models can use them

    :param df: Pandas DataFrame containing the dataset
    :return: Processed DataFrame with standardized column names and only necessary columns
    """
    if df is None:
        print("No data to process!")
        return None

    # Identifying col dynamically
    column_mapping = {
        "review_text": fuzzy_match_column(df, "review_text"),
        "rating": fuzzy_match_column(df, "rating")
    }

    #failure condition just in case we cant find any matches
    if not column_mapping["review_text"]:
        print("No suitable text column found for reviews.")
        print("Available columns:", df.columns.tolist())
        return None

    if not column_mapping["rating"]:
        print("No rating column found. Proceeding without ratings.")

    # Extracting and copying relvant colms
    selected_columns = {key: col for key, col in column_mapping.items() if col}
    df_cleaned = df[list(selected_columns.values())].copy()

    # Reverse mapping the keys
    reverse_mapping = {v: k for k, v in selected_columns.items()}
    df_cleaned.rename(columns=reverse_mapping, inplace=True)
    print("Renamed Columns:", df_cleaned.columns.tolist())

    # Dropping missing values in 'review_text' cause that would be bad if we send them in
    if "review_text" in df_cleaned.columns:
        df_cleaned.dropna(subset=["review_text"], inplace=True)
    else:
        print("'review_text' column not found after renaming!")

    if "rating" in df_cleaned.columns:
        df_cleaned["rating"] = df_cleaned["rating"].apply(extract_numeric_rating)
        df_cleaned.dropna(subset=["rating"], inplace=True)
        # Convert rating to int if possible
        df_cleaned["rating"] = df_cleaned["rating"].astype(int)

    print(f"Review Text Column: {column_mapping['review_text']}")
    print(f"Rating Column: {column_mapping['rating'] if column_mapping['rating'] else 'Not Found'}")
    print("\nFirst 5 rows of processed data:\n", df_cleaned.head())

    return df_cleaned
