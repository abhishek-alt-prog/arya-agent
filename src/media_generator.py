"""
Media Generator — Handles local generation of Images (Stable Diffusion) and Audio (Edge TTS).
"""
import os
import asyncio
import logging
from uuid import uuid4

# Setup logging
logger = logging.getLogger(__name__)

# Output directory for BFF static assets
ASSETS_DIR = "/Users/abhishek/Code/Arya/bff/src/main/resources/static/assets"
BFF_ASSETS_URL_PREFIX = "/assets"

# Create the directory if it doesn't exist
os.makedirs(ASSETS_DIR, exist_ok=True)

class MediaGenerator:
    def __init__(self):
        self.image_pipeline = None

    def _load_image_pipeline(self):
        """Lazy load the Stable Diffusion pipeline to save memory."""
        if self.image_pipeline is None:
            logger.info("Loading Stable Diffusion pipeline...")
            from diffusers import AutoPipelineForText2Image
            import torch
            # Use CPU or MPS (Apple Silicon) if available
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            # SD 1.5 is small and fast enough for local development
            self.image_pipeline = AutoPipelineForText2Image.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
                safety_checker=None # avoid unnecessary memory usage
            )
            self.image_pipeline = self.image_pipeline.to(device)
            # Reduce memory usage
            self.image_pipeline.enable_attention_slicing()
            logger.info("Stable Diffusion pipeline loaded on %s.", device)

    def generate_image(self, prompt: str, prefix: str = "img") -> str:
        """
        Generates an image from a text prompt and saves it to the static assets folder.
        Returns the URL path to the generated image.
        """
        self._load_image_pipeline()
        logger.info("Generating image for prompt: %s", prompt)
        
        # Add some positive keywords for kid-friendly illustrations
        full_prompt = f"colorful cartoon illustration for children, {prompt}, high quality, cute"
        
        # Generate
        image = self.image_pipeline(full_prompt, num_inference_steps=20).images[0]
        
        # Save
        filename = f"{prefix}_{uuid4().hex[:8]}.png"
        filepath = os.path.join(ASSETS_DIR, filename)
        image.save(filepath)
        
        logger.info("Saved image to %s", filepath)
        return f"{BFF_ASSETS_URL_PREFIX}/{filename}"
        
    def generate_audio(self, text: str, prefix: str = "audio") -> str:
        """
        Generates speech from text using edge-tts and saves it to the static assets folder.
        Returns the URL path to the generated audio.
        """
        if not text:
            return None
            
        logger.info("Generating audio for text (length %d)...", len(text))
        import edge_tts
        
        filename = f"{prefix}_{uuid4().hex[:8]}.mp3"
        filepath = os.path.join(ASSETS_DIR, filename)
        
        # Voice selection: "en-US-AriaNeural" or "en-GB-RyanNeural" or similar
        voice = "en-US-AriaNeural" # Fitting for Arya!
        
        async def _generate():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(filepath)
            
        # Run async in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                loop.run_until_complete(_generate())
            else:
                loop.run_until_complete(_generate())
        except RuntimeError:
            # If no event loop
            asyncio.run(_generate())
        except Exception as e:
            logger.error("Audio generation failed: %s", e)
            return None
            
        logger.info("Saved audio to %s", filepath)
        return f"{BFF_ASSETS_URL_PREFIX}/{filename}"

    def unload_models(self):
        """Free up memory."""
        if self.image_pipeline is not None:
            del self.image_pipeline
            self.image_pipeline = None
            logger.info("Unloaded Stable Diffusion pipeline.")
