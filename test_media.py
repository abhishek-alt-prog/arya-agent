from src.media_generator import MediaGenerator
print("Starting media test")
gen = MediaGenerator()
audio = gen.generate_audio("Hello, my name is Arya and I am learning about science.")
print("Audio:", audio)
image = gen.generate_image(
    "A flower with each part highlighted in a different color, "
    "pink petals, green stem, brown roots, arrows pointing to each part"
)
print("Image:", image)
gen.unload_models()
print("Done")
