import torch
from datasets import load_dataset
import os

print("Loading voice dataset...")


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


SAVE_PATH = os.path.join(CURRENT_DIR, "models", "speaker_speecht5.pt")

try:
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors",
        split="validation",
        revision="refs/convert/parquet"
    )

    print("Dataset loaded successfully from Parquet revision.")

    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    torch.save(speaker_embeddings, SAVE_PATH)
    print(f"SUCCESS: Created speaker embedding file at: {SAVE_PATH}")

except Exception as e:
    print(f"WARNING: Failed to download dataset ({e}).")
    print("Attempting to generate a random speaker embedding as a fallback...")

    speaker_embeddings = torch.zeros(1, 512)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    torch.save(speaker_embeddings, SAVE_PATH)
    print(f"SUCCESS: Created RANDOM speaker embedding file at: {SAVE_PATH}")
    print("NOTE: The voice will sound monotonic because this is a random vector.")