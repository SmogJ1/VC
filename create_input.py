import os
import json

image_dir = "images"
input_json = "data/input.json"

data = {
    "image_files": [("images/"+f) for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
}

with open(input_json, "w") as f:
    json.dump(data, f, indent=4)
