from huggingface_hub import HfApi, login
import os

# Login to Hugging Face (you'll be prompted for token)
login()

# Initialize the API
api = HfApi()

# Create repository with your actual username
repo_name = "Dombara/pii-detection-model"

try:
    # Create the repository
    api.create_repo(repo_id=repo_name, exist_ok=True)
    print(f"Repository {repo_name} created successfully!")
except Exception as e:
    print(f"Repository might already exist: {e}")

# Upload model files
model_path = "./deberta_pii_refinetuned/final/"

# List of files to upload
files_to_upload = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json"
]

for file_name in files_to_upload:
    file_path = os.path.join(model_path, file_name)
    if os.path.exists(file_path):
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=repo_name,
            )
            print(f"Uploaded {file_name} successfully!")
        except Exception as e:
            print(f"Error uploading {file_name}: {e}")
    else:
        print(f"File {file_name} not found in {model_path}")

print("Upload complete!")