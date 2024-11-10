import json
from utils import GoEmotionConfig
from gpt2 import EmotionDetector

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
        
        # Initialize the score
        score = 0.0
        counter = 0

        # Calculate the score based on emotion categories
        for i, prob in enumerate(probs_over_all28_emotions):
            emotion = self.emotions[i]  # Map index to emotion name

            if emotion in self.positive_emotions:
                score += prob.item()  # Positive contribution
                counter += 1
            elif emotion in self.negative_emotions:
                score -= prob.item()  # Negative contribution
                counter += 1
            # Neutral emotions do not affect the score

        return round(score, 4) / counter # Return rounded score for clarity

"""
# Usage Example
YEARLY_LYRICS = "HERE THE LYRICS OF CORRESPONDING YEAR SHOULD COME"
calculator = YearlyEmotionScoreCalculator()
yearly_score = calculator.calculate_emotion_score(YEARLY_LYRICS)
print(f"Yearly Emotion Score: {yearly_score}")
"""