from sentence_transformers import SentenceTransformer
import joblib

print("Loading model bundle...")
bundle = joblib.load("viral_detector_model (1).pkl")
model_name = bundle["embed_model_name"]

print(f"Downloading {model_name}...")
model = SentenceTransformer(model_name)
print("Model downloaded successfully!")
print(f"Model cached at: {model._model_card_data.model_name}")
