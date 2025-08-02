from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import json
import os
import time
import logging
import re
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variable
pii_pipe = None

# Enhanced PII patterns with better name detection
PII_PATTERNS = {
    'EMAIL': [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    ],
    'PHONE': [
        r'\+91[-\s]?\d{10}',
        r'\b\d{10}\b',
        r'\b\d{5}[-\s]?\d{5}\b',
        r'\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b',
        r'\b\d{4}[-\s]?\d{6}\b'
    ],
    'AADHAAR': [
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        r'\b\d{12}\b'
    ],
    'PAN': [
        r'\b[A-Z]{5}\d{4}[A-Z]\b'
    ],
    'CREDIT_CARD': [
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        r'\b\d{16}\b'
    ],
    'NAME': [
        # Names with titles (Dr., Prof., Mr., etc.)
        r'\b(?:Dr\.?|Prof\.?|Mr\.?|Mrs\.?|Ms\.?|Miss\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',
        # Regular full names (First Last, First Middle Last)
        r'\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?\b',
        # Three or more word names
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'
    ]
}

def load_model():
    """Load the fine-tuned PII detection model from Hugging Face Hub"""
    global pii_pipe
    
    # Your Hugging Face model repository
    model_name = "Dombara/pii-detection-model"
    
    try:
        logger.info(f"Loading model from Hugging Face Hub: {model_name}")
        
        # Load model directly from Hugging Face Hub
        pii_pipe = pipeline(
            "token-classification",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",
            device=-1  # Use CPU (-1) for compatibility
        )
        
        logger.info("✅ Model loaded successfully from Hugging Face Hub")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model from Hugging Face Hub: {str(e)}")
        logger.info("💡 Make sure the model is uploaded to Hugging Face and is public")
        raise

def extract_entities_with_patterns(text: str) -> List[Dict[str, Any]]:
    """Extract PII entities using enhanced pattern matching"""
    entities = []
    
    logger.info(f"🔍 Searching for patterns in: '{text}'")
    
    for entity_type, patterns in PII_PATTERNS.items():
        logger.info(f"🔍 Checking {entity_type} patterns...")
        
        for i, pattern in enumerate(patterns):
            logger.info(f"  Pattern {i+1}: {pattern}")
            
            for match in re.finditer(pattern, text):
                matched_text = match.group().strip()
                logger.info(f"    ✅ Found match: '{matched_text}' at {match.start()}-{match.end()}")
                
                # Additional validation for names
                if entity_type == 'NAME':
                    if not is_valid_name(matched_text):
                        logger.info(f"    ❌ Rejected name: '{matched_text}' (failed validation)")
                        continue
                    logger.info(f"    ✅ Accepted name: '{matched_text}'")
                
                entities.append({
                    'entity_type': entity_type,
                    'text': matched_text,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.95,  # High confidence for pattern matches
                    'source': 'pattern'
                })
    
    logger.info(f"📊 Pattern matching found {len(entities)} entities")
    return entities

def is_valid_name(name: str) -> bool:
    """Enhanced name validation"""
    name_lower = name.lower().strip()
    
    logger.info(f"🔍 Validating name: '{name}'")
    
    # Skip empty or very short names
    if len(name.strip()) < 3:
        logger.info("  ❌ Too short")
        return False
    
    # Skip common false positives
    false_positives = [
        'team members', 'send reports', 'backup to', 'contact me',
        'the meeting', 'new employee', 'customer service', 'email thread',
        'attended the', 'the company', 'our team', 'business report'
    ]
    
    if any(fp in name_lower for fp in false_positives):
        logger.info(f"  ❌ Contains false positive phrase")
        return False
    
    # Check if it has typical name characteristics
    words = name.split()
    if len(words) < 2:
        logger.info(f"  ❌ Less than 2 words")
        return False
    
    # Handle titles
    first_word = words[0].lower().rstrip('.')
    if first_word in ['dr', 'prof', 'mr', 'mrs', 'ms', 'miss']:
        # For titles, check remaining words
        name_words = words[1:]
    else:
        name_words = words
    
    # Validate name words
    for word in name_words:
        clean_word = word.rstrip('.,!?')
        if len(clean_word) < 2:
            logger.info(f"  ❌ Word too short: '{clean_word}'")
            return False
        if not clean_word[0].isupper():
            logger.info(f"  ❌ Word doesn't start with capital: '{clean_word}'")
            return False
        if not clean_word[1:].islower():
            logger.info(f"  ❌ Word not properly capitalized: '{clean_word}'")
            return False
    
    logger.info(f"  ✅ Valid name")
    return True

def extract_entities_with_model(text: str, threshold: float) -> List[Dict[str, Any]]:
    """Extract PII entities using the model with advanced processing"""
    if not pii_pipe:
        return []
    
    try:
        logger.info(f"🤖 Running model on text: '{text}'")
        
        # Get raw model predictions
        raw_entities = pii_pipe(text)
        logger.info(f"🤖 Model returned {len(raw_entities)} raw predictions")
        
        # Log all raw predictions
        for i, entity in enumerate(raw_entities):
            logger.info(f"  Raw {i+1}: '{entity['word']}' -> {entity['entity_group']} (conf: {entity['score']:.3f})")
        
        # Group tokens that belong to the same entity
        grouped_entities = []
        current_entity = None
        
        for entity in sorted(raw_entities, key=lambda x: x['start']):
            if entity['score'] < threshold:
                logger.info(f"  ❌ Skipping '{entity['word']}' (confidence {entity['score']:.3f} < {threshold})")
                continue
                
            entity_type = entity['entity_group'].replace('B-', '').replace('I-', '')
            
            # Check if this continues the current entity
            if (current_entity and 
                current_entity['entity_type'] == entity_type and
                entity['start'] <= current_entity['end'] + 3):  # Allow small gaps
                
                # Extend current entity
                current_entity['end'] = entity['end']
                current_entity['confidence'] = max(current_entity['confidence'], entity['score'])
                logger.info(f"  ➕ Extended entity to include '{entity['word']}'")
                
            else:
                # Save previous entity
                if current_entity:
                    current_entity['text'] = text[current_entity['start']:current_entity['end']].strip()
                    grouped_entities.append(current_entity)
                    logger.info(f"  ✅ Completed entity: '{current_entity['text']}' ({current_entity['entity_type']})")
                
                # Start new entity
                current_entity = {
                    'entity_type': entity_type,
                    'start': entity['start'],
                    'end': entity['end'],
                    'confidence': entity['score'],
                    'source': 'model'
                }
                logger.info(f"  🆕 Started new entity: '{entity['word']}' ({entity_type})")
        
        # Don't forget the last entity
        if current_entity:
            current_entity['text'] = text[current_entity['start']:current_entity['end']].strip()
            grouped_entities.append(current_entity)
            logger.info(f"  ✅ Final entity: '{current_entity['text']}' ({current_entity['entity_type']})")
        
        logger.info(f"🤖 Model extraction found {len(grouped_entities)} entities")
        return grouped_entities
        
    except Exception as e:
        logger.error(f"Model extraction error: {str(e)}")
        return []

def merge_and_deduplicate_entities(model_entities: List[Dict], pattern_entities: List[Dict]) -> List[Dict[str, Any]]:
    """Merge model and pattern entities, removing duplicates and choosing best"""
    logger.info(f"🔄 Merging {len(model_entities)} model entities with {len(pattern_entities)} pattern entities")
    
    all_entities = model_entities + pattern_entities
    
    if not all_entities:
        return []
    
    # Sort by start position
    all_entities.sort(key=lambda x: x['start'])
    
    merged = []
    
    for entity in all_entities:
        logger.info(f"🔄 Processing entity: '{entity['text']}' ({entity['entity_type']}) from {entity['source']}")
        
        # Check for overlap with existing entities
        overlapped = False
        
        for i, existing in enumerate(merged):
            # Check if entities overlap
            if not (entity['end'] <= existing['start'] or entity['start'] >= existing['end']):
                overlapped = True
                
                # Choose the better entity (pattern > model for reliability)
                if (entity['source'] == 'pattern' or 
                    (entity['source'] == existing['source'] and entity['confidence'] > existing['confidence']) or
                    (entity['end'] - entity['start']) > (existing['end'] - existing['start'])):
                    merged[i] = entity
                    logger.info(f"  ✅ Replaced with better entity: '{entity['text']}'")
                else:
                    logger.info(f"  ❌ Keeping existing entity: '{existing['text']}'")
                break
        
        if not overlapped:
            merged.append(entity)
            logger.info(f"  ➕ Added new entity: '{entity['text']}'")
    
    logger.info(f"📊 Merged entities count: {len(merged)}")
    return merged

def post_process_entities(entities: List[Dict], text: str) -> List[Dict[str, Any]]:
    """Post-process entities to fix boundaries and improve accuracy"""
    processed = []
    
    for entity in entities:
        entity_type = entity['entity_type']
        start = entity['start']
        end = entity['end']
        
        # Smart boundary adjustment for names
        if entity_type == 'NAME':
            # Extend to capture titles at the beginning
            extended_start = start
            while extended_start > 0:
                char = text[extended_start - 1]
                if char == ' ':
                    # Check if previous word is a title
                    word_start = extended_start - 1
                    while word_start > 0 and text[word_start - 1] not in ' \t\n':
                        word_start -= 1
                    word = text[word_start:extended_start - 1].strip()
                    if word.lower() in ['dr', 'prof', 'mr', 'mrs', 'ms', 'miss'] or word.endswith('.'):
                        extended_start = word_start
                    else:
                        break
                elif char.isalpha() or char == '.':
                    extended_start -= 1
                else:
                    break
            
            # Clean start boundary
            while extended_start < len(text) and text[extended_start] in ' \t':
                extended_start += 1
            
            entity['start'] = extended_start
            entity['text'] = text[extended_start:end].strip()
        
        # Validate entity makes sense
        if len(entity['text'].strip()) > 0:
            processed.append(entity)
    
    return processed

def detect_pii(text: str, threshold: float = 0.5) -> Dict[str, Any]:
    """Hybrid PII detection using both model and patterns"""
    start_time = time.time()
    
    try:
        # Extract entities using both methods
        model_entities = extract_entities_with_model(text, threshold)
        pattern_entities = extract_entities_with_patterns(text)
        
        logger.info(f"Model found {len(model_entities)} entities, patterns found {len(pattern_entities)} entities")
        
        # Merge and deduplicate
        merged_entities = merge_and_deduplicate_entities(model_entities, pattern_entities)
        
        # Post-process for better accuracy
        final_entities = post_process_entities(merged_entities, text)
        
        processing_time = time.time() - start_time
        
        # Log results for debugging
        for entity in final_entities:
            logger.info(f"Detected {entity['entity_type']}: '{entity['text']}' (confidence: {entity['confidence']:.3f}, source: {entity['source']})")
        
        return {
            'entities': final_entities,
            'total_entities': len(final_entities),
            'processing_time': round(processing_time, 4)
        }
        
    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        return {
            'entities': [],
            'total_entities': 0,
            'processing_time': 0.0
        }

def mask_pii_text(text: str, threshold: float = 0.5, mask_style: str = "asterisk") -> Dict[str, Any]:
    """Mask PII entities with improved accuracy"""
    detection_result = detect_pii(text, threshold)
    entities = detection_result['entities']
    
    if not entities:
        return {
            'original_text': text,
            'masked_text': text,
            'entities_masked': [],
            'summary': {
                'total_entities_masked': 0,
                'processing_time': detection_result['processing_time']
            }
        }
    
    # Sort entities by position (reverse order for masking)
    entities_sorted = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    masked_text = text
    masked_entities = []
    
    for entity in entities_sorted:
        start = entity['start']
        end = entity['end']
        original_text = entity['text']
        
        # Generate replacement based on mask style
        if mask_style == "type":
            replacement = f"[{entity['entity_type']}]"
        elif mask_style == "asterisk":
            replacement = "*" * len(original_text)
        elif mask_style == "partial":
            if len(original_text) <= 3:
                replacement = "*" * len(original_text)
            else:
                replacement = original_text[0] + "*" * (len(original_text) - 2) + original_text[-1]
        elif mask_style == "hash":
            replacement = "#" * len(original_text)
        elif mask_style == "redacted":
            replacement = "[REDACTED]"
        else:
            replacement = "*" * len(original_text)
        
        # Apply masking
        masked_text = masked_text[:start] + replacement + masked_text[end:]
        
        # Record masked entity
        masked_entities.append({
            'type': entity['entity_type'],
            'original_value': original_text,
            'masked_value': replacement,
            'confidence': f"{entity['confidence']:.1%}",
            'source': entity['source'],
            'position': {
                'start': start,
                'end': end
            }
        })
    
    return {
        'original_text': text,
        'masked_text': masked_text,
        'entities_masked': masked_entities,
        'summary': {
            'total_entities_masked': len(masked_entities),
            'processing_time': detection_result['processing_time']
        }
    }

# API Endpoints

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'message': 'PII Detection API is running',
        'model_source': 'Hugging Face Hub: Dombara/pii-detection-model',
        'endpoints': {
            'detect': 'POST /detect - Detect PII entities',
            'mask': 'POST /mask - Mask PII entities'
        }
    })

@app.route('/detect', methods=['POST'])
def detect_endpoint():
    """Hybrid PII detection endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: text'
            }), 400
        
        text = data['text']
        threshold = data.get('threshold', 0.5)
        
        # Basic validation
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({
                'success': False,
                'error': 'Text must be a non-empty string'
            }), 400
        
        if not 0 <= threshold <= 1:
            return jsonify({
                'success': False,
                'error': 'Threshold must be between 0 and 1'
            }), 400
        
        # Detect PII
        result = detect_pii(text, threshold)
        
        # Format response
        formatted_entities = []
        for entity in result['entities']:
            formatted_entities.append({
                'type': entity['entity_type'],
                'value': entity['text'],
                'confidence': f"{entity['confidence']:.1%}",
                'source': entity['source'],
                'position': {
                    'start': entity['start'],
                    'end': entity['end']
                }
            })
        
        return jsonify({
            'success': True,
            'input': {
                'text': text,
                'threshold': threshold
            },
            'results': {
                'entities_detected': formatted_entities,
                'total_entities': result['total_entities'],
                'processing_time': f"{result['processing_time']:.3f}s"
            }
        })
        
    except Exception as e:
        logger.error(f"Error in detect endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/mask', methods=['POST'])
def mask_endpoint():
    """Hybrid PII masking endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: text'
            }), 400
        
        text = data['text']
        threshold = data.get('threshold', 0.5)
        mask_style = data.get('mask_style', 'asterisk')
        
        # Basic validation
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({
                'success': False,
                'error': 'Text must be a non-empty string'
            }), 400
        
        if not 0 <= threshold <= 1:
            return jsonify({
                'success': False,
                'error': 'Threshold must be between 0 and 1'
            }), 400
        
        valid_mask_styles = ['type', 'asterisk', 'partial', 'hash', 'redacted']
        if mask_style not in valid_mask_styles:
            return jsonify({
                'success': False,
                'error': f'Invalid mask style. Available: {", ".join(valid_mask_styles)}'
            }), 400
        
        # Mask PII
        result = mask_pii_text(text, threshold, mask_style)
        
        return jsonify({
            'success': True,
            'input': {
                'text': text,
                'threshold': threshold,
                'mask_style': mask_style
            },
            'results': {
                'original_text': result['original_text'],
                'masked_text': result['masked_text'],
                'entities_masked': result['entities_masked'],
                'summary': {
                    'total_entities_masked': result['summary']['total_entities_masked'],
                    'processing_time': f"{result['summary']['processing_time']:.3f}s"
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in mask endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET / - Health check and API info',
            'POST /detect - Detect PII entities',
            'POST /mask - Mask PII entities'
        ]
    }), 404

if __name__ == '__main__':
    try:
        print("🚀 Starting Hybrid PII Detection API...")
        print("📦 Model Source: Hugging Face Hub (Dombara/pii-detection-model)")
        load_model()
        print("✅ Model loaded successfully!")
        print("\n🎯 Detection Strategy:")
        print("  • Model-based detection for context understanding")
        print("  • Pattern-based detection for reliability")
        print("  • Smart merging and deduplication")
        print("  • Post-processing for accuracy")
        print("\n🌐 Server running on: http://localhost:5000")
        
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        print(f"❌ Failed to start API: {e}")