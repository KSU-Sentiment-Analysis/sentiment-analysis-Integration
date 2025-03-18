import math
import json
from transformers import pipeline


def load_advanced_models():
    #loading pre-build models to see how it goes
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=0  # Use GPU (cuda:0) or set to -1 if u dont have dedicated gpu
    )
    print("✅ Advanced sentiment model loaded: cardiffnlp/twitter-roberta-base-sentiment")

    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=0  # Use GPU (cuda:0) or set to -1 if u dont have dedicated gpu
    )
    print("✅ Emotion detection model loaded: j-hartmann/emotion-english-distilroberta-base")

    return sentiment_model, emotion_model


def detect_sarcasm(text):
    """
    A simple method need to make this better later. currently it just doesnt work at all
    """
    sarcasm_indicators = ["yeah right", "as if", "oh great", "just what i needed", "sure, why not"]
    text_lower = text.lower()
    for phrase in sarcasm_indicators:
        if phrase in text_lower:
            return True
    return False


def aspect_analysis(text):
    """
    Aspect based analyussis simple but it kind works!
    """
    aspects = {}
    aspect_keywords = {
        "price": ["price", "cost", "expensive", "cheap"],
        "quality": ["quality", "durable", "well-made", "poor"],
        "shipping": ["shipping", "delivery", "fast", "slow"],
        "customer service": ["customer service", "support", "helpful", "rude"]
    }
    text_lower = text.lower()
    for aspect, keywords in aspect_keywords.items():
        count = sum(text_lower.count(word) for word in keywords)
        if count > 0:
            # Mark as positive if count>=2, else neutral.
            aspects[aspect] = "positive" if count >= 2 else "neutral"
    return aspects


def predict_sentiment(texts, sentiment_model, batch_size=32):
    """
    calls our sentiment model need to tweak (batches)
    """
    return sentiment_model(texts, truncation=True, max_length=512, batch_size=batch_size)


def predict_emotion(texts, emotion_model, batch_size=32):
    """
    calls our emotion model need to tweak (batches)
    """
    return emotion_model(texts, truncation=True, max_length=512, batch_size=batch_size)


def analyze_reviews_batch(reviews, batch_size=32):
    """
    Processes reviews in batches for advanced sentiment analysis.

    use batch processing for emotin and sarc and sentiment

    Returns a list of dictionaries (one per review) with keys:
      - advanced_sentiment
      - advanced_sentiment_score
      - advanced_emotion (dict)
      - sarcasm (bool)
      - aspects (dict)
    """
    sentiment_model, emotion_model = load_advanced_models()
    results = []
    total = len(reviews)
    num_batches = math.ceil(total / batch_size)

    for i in range(num_batches):
        batch = reviews[i * batch_size: (i + 1) * batch_size]
        sentiment_results = predict_sentiment(batch, sentiment_model, batch_size=batch_size)
        emotion_results = predict_emotion(batch, emotion_model, batch_size=batch_size)

        for j, text in enumerate(batch):
            sent_result = sentiment_results[j]
            adv_sentiment = sent_result["label"]
            adv_sentiment_score = sent_result["score"]
            emo_result = emotion_results[j]
            emotion_scores = {entry["label"]: entry["score"] for entry in emo_result}
            sarcasm_flag = detect_sarcasm(text)
            aspects = aspect_analysis(text)

            results.append({
                "advanced_sentiment": adv_sentiment,
                "advanced_sentiment_score": adv_sentiment_score,
                "advanced_emotion": emotion_scores,
                "sarcasm": sarcasm_flag,
                "aspects": aspects
            })
        print(f"Processed batch {i + 1} of {num_batches}")
    return results


def analyze_reviews(reviews, batch_size=32):

    return analyze_reviews_batch(reviews, batch_size)
