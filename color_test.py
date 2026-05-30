from PIL import Image
import numpy as np
from src.color_space import rgb_to_ycbcr, ycbcr_to_rgb

# Load image
img = np.array(Image.open("images/lena_synth.png").convert("RGB"))

# Convert RGB → YCbCr
ycbcr = rgb_to_ycbcr(img)

# Convert back YCbCr → RGB
rgb_back = ycbcr_to_rgb(ycbcr)

# Save results
Image.fromarray(img).save("output/original.png")
Image.fromarray(np.clip(ycbcr, 0, 255).astype("uint8")).save("output/ycbcr_as_rgb.png")
Image.fromarray(np.clip(rgb_back, 0, 255).astype("uint8")).save("output/reconstructed.png")

print("Done! Check output folder.")