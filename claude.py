import boto3
import json

# Define the prompt
prompt_data = "Act as a Shakespearean and write a poem on machine learning."

# Create Bedrock runtime client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Build payload for Claude Text
payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "temperature": 0.5,
    "top_p": 0.8,
    "messages": [
        {"role": "user", "content": prompt_data}
    ]
}

# Convert payload to JSON
body = json.dumps(payload)

# Model ID for Claude Text Lite
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# Invoke the model
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

# Parse response
response_body = json.loads(response["body"].read())

# Titan responses are in "results"
response_text = response_body["content"][0]["text"]

# Print result
print(response_text)
