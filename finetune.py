import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import torch
import json
import os
import warnings

# FORCE GPU USAGE - FAIL IF NO GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)
else:
    print("❌ CRITICAL ERROR: GPU NOT AVAILABLE!")
    print("This script REQUIRES GPU training. Exiting...")
    exit(1)

# Verify GPU with more detailed checks
print("=== MANDATORY GPU STATUS CHECK ===")
if not torch.cuda.is_available():
    print("❌ FATAL ERROR: CUDA not available!")
    exit(1)

device = torch.device("cuda:0")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
print(f"✅ CUDA Version: {torch.version.cuda}")
print(f"✅ PyTorch Version: {torch.__version__}")
print(f"✅ Device: {device}")
print(f"✅ GPU Count: {torch.cuda.device_count()}")
print(f"✅ Current GPU: {torch.cuda.current_device()}")
print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
print(f"✅ GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"✅ GPU Memory Free: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
print("==================================\n")

# Load CSV data
df = pd.read_csv("pii_dataset.csv")

# Prepare tokens and labels
sentences = []
labels = []
for _, row in df.iterrows():
    tokens = row["text"].split()
    tags = row["labels"].split()
    if len(tokens) == len(tags):
        sentences.append(tokens)
        labels.append(tags)

# Create dataset
dataset = Dataset.from_dict({"tokens": sentences, "ner_tags": labels})

# 🔧 FIXED: Force safetensors loading with FP32 to match training args
model_name = "lakshyakh93/deberta_finetuned_pii"
print(f"Loading model: {model_name}")

try:
    print("🔄 Attempting to load with safetensors format (FP32)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Changed from float16 to float32
        use_safetensors=True,
        trust_remote_code=True
    )
    print("✅ Model loaded successfully with safetensors (FP32)!")
    
except Exception as e:
    print(f"❌ Safetensors loading failed: {str(e)}")
    print("🔄 Trying to download and convert to safetensors...")
    
    try:
        # Download model files manually and convert
        from huggingface_hub import snapshot_download
        import tempfile
        
        # Download to temp directory
        temp_dir = tempfile.mkdtemp()
        snapshot_download(
            repo_id=model_name,
            local_dir=temp_dir,
            allow_patterns=["*.safetensors", "config.json", "tokenizer*"]
        )
        
        tokenizer = AutoTokenizer.from_pretrained(temp_dir)
        model = AutoModelForTokenClassification.from_pretrained(
            temp_dir,
            torch_dtype=torch.float32,  # Changed from float16 to float32
            use_safetensors=True
        )
        print("✅ Model loaded from downloaded safetensors (FP32)!")
        
    except Exception as e2:
        print(f"❌ Safetensors download failed: {str(e2)}")
        print("🔄 Using base DeBERTa model as fallback...")
        
        try:
            # Fallback to base model
            base_model = "microsoft/deberta-v3-base" 
            print(f"Loading base model: {base_model}")
            
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForTokenClassification.from_pretrained(
                base_model,
                num_labels=50,  # Will be resized anyway
                torch_dtype=torch.float32  # Changed from float16 to float32
            )
            print("✅ Base DeBERTa model loaded successfully (FP32)!")
            
        except Exception as e3:
            print(f"❌ All loading methods failed: {str(e3)}")
            print("\n🚨 CRITICAL ERROR: Cannot load any model!")
            print("💡 SOLUTIONS:")
            print("1. Upgrade PyTorch: pip install torch>=2.6.0 --index-url https://download.pytorch.org/whl/cu121")
            print("2. Use a different model that has safetensors format")
            print("3. Downgrade transformers: pip install transformers==4.35.0")
            exit(1)

# FORCE MODEL TO GPU - CRITICAL
print(f"🔄 Moving model to GPU: {device}")
model = model.to(device)
model_device = next(model.parameters()).device
print(f"✅ Model successfully moved to: {model_device}")

# VERIFY MODEL IS ON GPU
if model_device.type != 'cuda':
    print(f"❌ CRITICAL ERROR: Model is on {model_device}, not GPU!")
    print("Training ABORTED - Model must be on GPU!")
    exit(1)
else:
    print(f"✅ CONFIRMED: Model is on GPU {model_device}")
    print(f"✅ GPU Memory after model load: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")

# Get original label mappings
original_label2id = model.config.label2id
original_id2label = model.config.id2label
print(f"Original labels in the model: {len(original_label2id)}")
print(f"Original labels: {list(original_label2id.keys())[:10]}...")  # Show first 10

# Get unique labels from dataset
unique_labels = set()
for label_seq in labels:
    unique_labels.update(label_seq)
print(f"Labels in your dataset: {len(unique_labels)}")
print(f"Dataset labels: {sorted(unique_labels)}")

# Find and add new labels
new_labels = unique_labels - set(original_label2id.keys())
print(f"New labels to be added: {len(new_labels)} - {list(new_labels)}")

extended_label2id = original_label2id.copy()
extended_id2label = original_id2label.copy()
for new_label in new_labels:
    new_id = len(extended_label2id)
    extended_label2id[new_label] = new_id
    extended_id2label[new_id] = new_label

print(f"Extended label vocabulary size: {len(extended_label2id)}")

# Resize classification head
old_num_labels = len(original_label2id)
new_num_labels = len(extended_label2id)
if new_num_labels > old_num_labels:
    print(f"🔧 Resizing classification head from {old_num_labels} to {new_num_labels} labels")
    old_classifier_weight = model.classifier.weight.data.clone()
    old_classifier_bias = model.classifier.bias.data.clone()
    
    # Create new classifier and FORCE to GPU
    model.classifier = torch.nn.Linear(model.classifier.in_features, new_num_labels)
    model.classifier = model.classifier.to(device)
    print(f"✅ New classifier moved to: {next(model.classifier.parameters()).device}")
    
    with torch.no_grad():
        model.classifier.weight.data[:old_num_labels] = old_classifier_weight
        model.classifier.bias.data[:old_num_labels] = old_classifier_bias
        torch.nn.init.normal_(model.classifier.weight.data[old_num_labels:], mean=0.0, std=0.02)
        torch.nn.init.zeros_(model.classifier.bias.data[old_num_labels:])

# Update model configuration
model.config.label2id = extended_label2id
model.config.id2label = extended_id2label
model.config.num_labels = new_num_labels
model.num_labels = new_num_labels

print(f"Model num_labels updated to: {model.num_labels}")

# Map string labels to IDs
def encode_labels(example):
    example["labels"] = [extended_label2id[label] for label in example["ner_tags"]]
    return example

dataset = dataset.map(encode_labels)

# Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=False,
        max_length=256,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
    )

    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(labels[word_idx] if word_idx < len(labels) else -100)
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx
        all_labels.append(aligned_labels)
    
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

print("🔄 Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_and_align_labels, 
    batched=True,
    remove_columns=["tokens", "ner_tags", "labels"],
    desc="Tokenizing"
)

# Split dataset
train_size = int(0.8 * len(tokenized_dataset))
train_dataset = tokenized_dataset.select(range(train_size))
eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
print(f"📊 Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")

# Define metrics
seqeval = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)
    true_predictions = [[extended_id2label[p] for (p, l) in zip(pred, lab) if l != -100] 
                       for pred, lab in zip(predictions, labels)]
    true_labels = [[extended_id2label[l] for (p, l) in zip(pred, lab) if l != -100] 
                  for pred, lab in zip(predictions, labels)]
    return seqeval.compute(predictions=true_predictions, references=true_labels)

# Training arguments optimized for RTX 4050 - FIXED FP16 issue
training_args = TrainingArguments(
    output_dir="./deberta_pii_refinetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",  # Changed from "eval_f1" to "eval_overall_f1"
    greater_is_better=True,
    logging_dir="./logs",
    logging_steps=500,
    warmup_steps=500,
    dataloader_pin_memory=True,
    save_total_limit=1,
    report_to=None,
    dataloader_num_workers=0,
    fp16=False,
    bf16=False,
    dataloader_drop_last=False,
    remove_unused_columns=True,
    label_smoothing_factor=0.0,
    gradient_checkpointing=False,
    max_grad_norm=1.0,
    eval_accumulation_steps=4,
    save_safetensors=True,
    no_cuda=False,
)

# Data collator
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    max_length=256,
    pad_to_multiple_of=8,
    return_tensors="pt"
)

# Initialize Trainer
print("🔄 Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,  # Changed from tokenizer= to processing_class=
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Clear GPU cache before training
torch.cuda.empty_cache()
print(f"🚀 GPU Memory before training: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")

# FINAL GPU VERIFICATION
print("\n=== FINAL GPU STATUS BEFORE TRAINING ===")
print(f"✅ Model device: {next(model.parameters()).device}")
print(f"✅ Training device: {'GPU (CUDA)' if not training_args.no_cuda else 'CPU'}")
print(f"✅ FP16 enabled: {training_args.fp16}")
print(f"✅ Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"✅ Model: {model_name}")
print("==========================================\n")

# Start training
print("🚀 STARTING GPU TRAINING...")
print(f"🎯 Training device: {next(model.parameters()).device}")
print(f"🎯 Using: {torch.cuda.get_device_name(0)}")

try:
    trainer.train()
    print("✅ GPU TRAINING COMPLETED SUCCESSFULLY!")
    
    # Save model
    output_dir = "./deberta_pii_refinetuned/final"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mappings
    with open(f"{output_dir}/label_mappings.json", "w") as f:
        json.dump({
            "label2id": extended_label2id,
            "id2label": {str(k): v for k, v in extended_id2label.items()}
        }, f, indent=2)
    
    print(f"✅ Model saved with {len(extended_label2id)} labels!")
    print(f"🆕 New Indian PII labels added: {list(new_labels)}")
    print(f"💾 Model saved to: {output_dir}")
    print(f"🎮 Final GPU Memory: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
        
except Exception as e:
    print(f"❌ GPU TRAINING FAILED: {str(e)}")
    print(f"💾 GPU Memory at error: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
    print("\n🔧 GPU Troubleshooting tips:")
    print("1. Reduce batch_size to 1 (already set)")
    print("2. Reduce max_length to 128")
    print("3. Disable gradient_checkpointing")
    print("4. Restart and clear GPU memory")
    raise e
    
finally:
    torch.cuda.empty_cache()
    print("🧹 GPU cache cleared")