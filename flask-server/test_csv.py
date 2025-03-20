import pandas as pd

# Load the CSV file
df = pd.read_csv("sentiment_analysis_results_advanced.csv")

# Filter rows where sarcasm_flag is True
sarcastic_reviews = df[df['sarcasm_flag'] == True]

print("Number of reviews flagged as sarcastic:", len(sarcastic_reviews))
print(sarcastic_reviews[['review_text', 'sarcasm_flag']])