from transformers import pipeline

class EmotionJournal:
    def __init__(self):
        print("Loading AI Emotion Model... Please wait.")
        self.emotion_model = pipeline("sentiment-analysis")

    def analyze_emotion(self, text):
        """Analyzes the mood of the given journal entry."""
        result = self.emotion_model(text)[0]
        mood = result['label']
        score = result['score']

        return {
            "mood": mood,
            "confidence": round(score, 2)
        }

    def generate_reflection(self, mood):
        """Generates supportive emotional reflections."""
        reflections = {
            "POSITIVE": "It's wonderful to see you're feeling good! Keep building on this positive energy.",
            "NEGATIVE": "It seems you're dealing with something heavy. Remember, it's okay to feel down—you're stronger than you think.",
            "NEUTRAL": "A balanced day can be grounding. Take a moment to reflect on small wins.",
        }
        return reflections.get(mood, "Every feeling is valid. Continue expressing your thoughts.")

# ---------------------------
# Example Usage
# ---------------------------

if __name__ == "__main__":
    journal = EmotionJournal()
    
    user_text = input("How are you feeling today?\n")
    analysis = journal.analyze_emotion(user_text)
    
    print("\n--- Emotion Analysis ---")
    print(f"Mood: {analysis['mood']}")
    print(f"Confidence: {analysis['confidence']}")

    print("\n--- Supportive Reflection ---")
    print(journal.generate_reflection(analysis['mood']))
