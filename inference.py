from utils import GoEmotionConfig
from gpt2 import EmotionDetector

config = GoEmotionConfig()
detector = EmotionDetector(config)
probs_over_all28_emotions = detector.inference("Hi, I am a emotion detector! I am very happy to meet you.", threshold=0.4)
print(probs_over_all28_emotions)