from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification

print("Loading AI Models... This may take a while...")

# Model Check Text (Toxic)
text_classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

# Model Check Image (NSFW)
image_processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
image_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")

print("AI Models Loaded Successfully!")