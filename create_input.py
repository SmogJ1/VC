import os
import json

image_dir = "images"
output_json = "data/input.json"

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]

input_data = []
for img in image_files:
    input_data.append({
        "image_path": os.path.join(image_dir, img)
    })

with open(output_json, "w") as f:
    json.dump(input_data, f, indent=4)

print(f"Input JSON saved to {output_json}")
