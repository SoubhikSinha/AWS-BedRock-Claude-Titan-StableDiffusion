import boto3
import json
import base64
import os

# Create Bedrock runtime client (must be in us-west-2 for Stability AI)
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

prompt_data = "4K ultra HD cinematic beach scene, blue sky, rainy season, dramatic lighting"

payload = {
    "prompt": prompt_data,
    "mode": "text-to-image",   # required for text → image
    "aspect_ratio": "1:1",     # can be "16:9", "9:16", etc.
    "output_format": "png"
}

body = json.dumps(payload)

# Use Stable Image Ultra model (best quality, supports on-demand)
model_id = "stability.stable-image-ultra-v1:0"

response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

# Parse response
response_body = json.loads(response["body"].read())

# Decode base64 image
img_base64 = response_body["images"][0]
img_bytes = base64.b64decode(img_base64)

# Save image
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
file_name = f"{output_dir}/generated-img.png"
with open(file_name, "wb") as f:
    f.write(img_bytes)

print("✅ Image saved at:", file_name)
