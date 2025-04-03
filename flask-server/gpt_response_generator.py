import os
import json
import pandas as pd
from openai import AzureOpenAI

endpoint = "https://ebarn-m8w2cx30-eastus2.cognitiveservices.azure.com/"
model_name = "gpt-4"
deployment = "gpt-4"

subscription_key = "37dbO16yaOyiC3J5yEskDZL9j4WfxhCcoCvFK6SUIC9JNJbp4jzLJQQJ99BCACHYHv6XJ3w3AAAAACOGOkCR"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

def get_brand_guidelines():
    return {
        "apology_template": "We're truly sorry to hear about your experience.",
        "thanks_template": "We genuinely appreciate your kind words!",
        "preferred_customer_reference": "valued customer",
        "brand_keywords": ["quality", "style", "comfort", "confidence"],
        "default_response": "We're always here to listen and assist."
    }

def get_primary_emotion(emotion_json: str) -> str:
    try:
        emotions = json.loads(emotion_json)
        return max(emotions, key=emotions.get)
    except:
        return "neutral"

def build_prompt(review: pd.Series, guidelines: dict) -> str:
    sentiment = review.get('advanced_sentiment', 'neutral')
    primary_emotion = get_primary_emotion(review.get('advanced_emotion', '{}'))
    rating = review.get('rating', 'N/A')
    sarcasm = "Yes" if review.get('sarcasm_flag', False) else "No"
    aspects = review.get('aspect_analysis', '{}')
    if aspects == '{}' or not aspects:
        aspects = 'No specific aspects mentioned.'
    instruction = "Provide a neutral, empathetic response addressing the customer's feelings."
    if primary_emotion.lower() in ['anger', 'sadness', 'disgust'] or sentiment.upper() == "LABEL_1":
        instruction = f"{guidelines['apology_template']} Offer specific help and reassurance."
    elif primary_emotion.lower() in ['joy', 'surprise'] or sentiment.upper() == "LABEL_2":
        instruction = f"{guidelines['thanks_template']} Reinforce their positive feelings."

    prompt = (
        f"Customer Review: '{review.get('review_text', '')}'\n"
        f"Rating: {rating}/5\n"
        f"Primary Emotion: {primary_emotion.capitalize()}\n"
        f"Sarcasm Detected: {sarcasm}\n"
        f"Aspects Mentioned: {aspects}\n"
        f"Instruction: {instruction}\n"
        f"Brand Voice: empathetic, stylish, reassuring, and personalized.\n"
        "Response:"
    )
    return prompt

# ✅ FIXED: Correct call using 'complete'
def call_gpt_model(prompt: str) -> str:
    try:

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You're an empathetic, stylish customer service representative who responds personally and professionally.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=200,
            temperature=1.0,
            top_p=1.0,
            model=deployment
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"We're experiencing an issue generating your response: {str(e)}"

def evaluate_response(response: str, guidelines: dict) -> float:
    score = 0.5
    for keyword in guidelines["brand_keywords"]:
        if keyword.lower() in response.lower():
            score += 0.125
    return min(score, 1.0)

def generate_structured_responses(df: pd.DataFrame, sample_size: int = 5):
    guidelines = get_brand_guidelines()
    sampled_df = df.sample(n=min(sample_size, len(df)))
    responses, evaluations, prompts = [], [], []

    for _, row in sampled_df.iterrows():
        prompt = build_prompt(row, guidelines)
        gpt_response = call_gpt_model(prompt)
        score = evaluate_response(gpt_response, guidelines)
        responses.append(gpt_response)
        evaluations.append(score)
        prompts.append(prompt)

    return responses, evaluations, prompts
