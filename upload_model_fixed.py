from huggingface_hub import HfApi, login, whoami
import os
import time

# Login to Hugging Face
login()

# Get username
user_info = whoami()
username = user_info['name']
print(f"✅ Logged in as: {username}")

# Initialize API
api = HfApi()
repo_name = f"{username}/pii-detection-model"

print(f"📦 Using repository: {repo_name}")
print("⚠️ Make sure the repository exists on HuggingFace first!")

# Check model files
model_path = "./deberta_pii_refinetuned/final/"
if os.path.exists(model_path):
    actual_files = os.listdir(model_path)
    print(f"📋 Found files: {actual_files}")
else:
    print(f"❌ Model path not found: {model_path}")
    exit(1)

# Upload ALL files found in your directory
files_to_upload = actual_files  # Use all actual files

uploaded_count = 0
for file_name in files_to_upload:
    file_path = os.path.join(model_path, file_name)
    try:
        print(f"📤 Uploading {file_name}...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_name,
            repo_id=repo_name,
            repo_type="model"
        )
        print(f"✅ Uploaded {file_name} successfully!")
        uploaded_count += 1
        time.sleep(0.5)  # Small delay
    except Exception as e:
        print(f"❌ Error uploading {file_name}: {e}")

print(f"\n🎉 Upload complete! Successfully uploaded {uploaded_count} files to {repo_name}")

# Upload README
readme_content = f"""---
license: apache-2.0
tags:
- token-classification
- pii-detection
- privacy
- named-entity-recognition
language:
- en
pipeline_tag: token-classification
---

# PII Detection Model

Fine-tuned DeBERTa model for detecting Personally Identifiable Information (PII) in text.

## Usage

```python
from transformers import pipeline

pii_pipe = pipeline(
    "token-classification",
    model="{repo_name}",
    aggregation_strategy="simple"
)

text = "Contact John Doe at john@example.com or 9876543210"
results = pii_pipe(text)
```

## Detected PII Types
- Names, Email addresses, Phone numbers
- Aadhaar numbers, PAN cards, Credit card numbers
"""

try:
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="model"
    )
    print("✅ README.md uploaded successfully!")
except Exception as e:
    print(f"❌ Error uploading README: {e}")

print(f"\n🌐 Check your model at: https://huggingface.co/{repo_name}")