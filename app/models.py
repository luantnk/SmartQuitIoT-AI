from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification

print("Loading AI Models... This may take a while...")


# We use a 'pipeline'. This is a high-level helper from Hugging Face.
# It automatically handles:
# 1. Tokenization (breaking text into numbers the model understands).
# 2. Running the model (Unitary's toxic-bert).
# 3. Post-processing (converting raw scores into readable labels).
# 'top_k=None' means we want to see scores for ALL labels (toxic, insult, threat, etc.),
# not just the top one.
text_classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)


# Image models require two parts:
# 1. The Processor: This prepares the image. It resizes it, crops it,
#    and adjusts colors (normalization) to match exactly what the AI expects.
# 2. The Model: This is the actual neural network that looks at the processed data.
image_processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
image_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")

print("AI Models Loaded Successfully!")