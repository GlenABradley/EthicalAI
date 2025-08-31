import os
from sentence_transformers import SentenceTransformer

# Set environment variables
os.environ["HF_HOME"] = "./models/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "./models/huggingface"

# Ensure directories exist
os.makedirs("./models/sentence-transformers_all-mpnet-base-v2", exist_ok=True)
os.makedirs("./models/huggingface", exist_ok=True)

print("Downloading model...")
try:
    # Download the model
    model = SentenceTransformer(
        'sentence-transformers/all-mpnet-base-v2',
        cache_folder='./models',
        device='cpu'
    )
    print("Model downloaded successfully!")
    print(f"Model saved to: {model._model_name_or_path}")
except Exception as e:
    print(f"Error downloading model: {e}")
    raise
