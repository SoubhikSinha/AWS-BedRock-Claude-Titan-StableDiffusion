import boto3
import json

# Define the prompt
prompt_data = "Act as a Shakespearean and write a poem on machine learning."

# Create Bedrock runtime client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Build payload for Titan Text
payload = {
    "inputText": prompt_data,
    "textGenerationConfig": {
        "temperature": 0.5,
        "topP": 0.9,
        "maxTokenCount": 512,
        "stopSequences": []
    }
}

# Convert payload to JSON
body = json.dumps(payload)

# Model ID for Titan Text Lite
model_id = "amazon.titan-text-lite-v1"

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
response_text = response_body["results"][0]["outputText"]

# Print result
print(response_text)
