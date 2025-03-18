import torch
from transformers import pipeline

# Automatically use GPU (device 0) if available, otherwise use CPU (-1) # TAKES A LONG TIME BTW 30mins
device = 0 if torch.cuda.is_available() else -1

def load_fast_sentiment_model():
    """
    Loads pre-trained sentiment analysis model
    """
    fast_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )
    print("distilbert-base-uncased-finetuned-sst-2-english")
    return fast_model

def load_accurate_sentiment_model():
    """
    loads the twitter one but it kinda works surpsingly even if its twitter
    """
    accurate_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=device
    )
    print("cardiffnlp/twitter-roberta-base-sentiment")
    return accurate_model
