import json
from utils import GoEmotionConfig
from gpt2 import EmotionDetector
import math

class EmotionScoreCalculator:
    def __init__(self):
        # Initialize model and config
        self.config = GoEmotionConfig()
        self.detector = EmotionDetector(self.config)
        
        # Load sentiment mapping for emotion categories
        with open("sentiment_mapping.json", "r") as f:
            sentiment_mapping = json.load(f)
        
        self.positive_emotions = sentiment_mapping["positive"]
        self.negative_emotions = sentiment_mapping["negative"]
        self.neutral_emotions = sentiment_mapping["ambiguous"]  # Not affecting the score

        # Emotion names associated with model predictions
        self.emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

    def calculate_emotion_score(self, lyrics):
        # Perform inference on the lyrics
        probs_over_all28_emotions = self.detector.inference(lyrics, threshold=0.4)
        
        # Initialize cumulative scores
        positive_score = 0.0
        negative_score = 0.0

        # Calculate the score based on emotion categories with weighted contributions
        for i, prob in enumerate(probs_over_all28_emotions):
            emotion = self.emotions[i]  # Map index to emotion name
            weight = 1.0  # Default weight

            # Assign higher weight for stronger probabilities
            if prob >= 0.60:
                weight = 1.50

            elif prob >= 0.50:
                weight = 1.25

            elif prob >= 0.40:
                weight = 1.0
                
            else:
                weight = 0.0

            if emotion in self.positive_emotions:
                positive_score += prob.item() * weight  # Positive contribution
            elif emotion in self.negative_emotions:
                negative_score += prob.item() * weight  # Negative contribution
            # Neutral emotions do not affect the score

        # Apply logarithmic transformation for scaling
        positive_score = math.log(positive_score + 1)
        negative_score = math.log(negative_score + 1)

        return round(positive_score, 4), round(negative_score, 4)