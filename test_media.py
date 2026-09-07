from src.media_generator import MediaGenerator
print("Starting media test")
gen = MediaGenerator()
audio = gen.generate_audio("Hello, my name is Arya and I am learning about science.")
print("Audio:", audio)
image = gen.generate_image(
    "A clear educational diagram of a flower with labels 'Petal', 'Stem', 'Leaf', and 'Roots' pointing to each part"
)
print("Image:", image)
gen.unload_models()
print("Done")
