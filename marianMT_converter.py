import os
import tensorflow as tf
import json
from transformers import MarianTokenizer, TFAutoModelForSeq2SeqLM

# Configuration
MODEL_PAIRS = {
    'en-fr': {
        'forward': "Helsinki-NLP/opus-mt-en-fr",
        'backward': "Helsinki-NLP/opus-mt-fr-en"
    }
}

SELECTED_PAIR = 'en-fr'
OUTPUT_DIR = "tflite_models"

def sanitize_model_name(model_name):
    return model_name.replace('/', '_').replace('-', '_')

def log_message(message):
    print(f"[INFO] {message}")

def download_marianmt_model(model_name):
    log_message(f"Downloading model and tokenizer: {model_name}")
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    log_message(f"Successfully downloaded model: {model_name}")
    return model, tokenizer

def save_tensorflow_model(model, model_name):
    sanitized_name = sanitize_model_name(model_name)
    tf_model_path = os.path.join("saved_models", sanitized_name)
    os.makedirs(tf_model_path, exist_ok=True)
    log_message(f"Saving TensorFlow model: {model_name} to {tf_model_path}")
    model.save_pretrained(tf_model_path)
    tf.saved_model.save(model, tf_model_path)
    log_message(f"TensorFlow model saved successfully: {model_name}")
    return tf_model_path

def convert_to_tflite(tf_model_path, output_dir, model_name):
    try:
        log_message(f"Converting TensorFlow model to TFLite: {model_name}")
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(output_dir, f"{sanitize_model_name(model_name)}.tflite")
        os.makedirs(output_dir, exist_ok=True)
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        
        log_message(f"TFLite model saved to {tflite_path}")
        return tflite_path
    except Exception as e:
        log_message(f"Error converting to TFLite: {e}")
        raise

def save_vocabulary(tokenizer, output_dir, model_name):
    sanitized_name = sanitize_model_name(model_name)
    vocab_dir = os.path.join(output_dir, 'vocab')
    os.makedirs(vocab_dir, exist_ok=True)
    
    log_message(f"Saving vocabulary and special tokens for model: {model_name}")
    vocab = tokenizer.get_vocab()
    vocab_path = os.path.join(vocab_dir, f"{sanitized_name}_vocab.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({'vocab': vocab}, f, ensure_ascii=False, indent=2)
    
    special_tokens = {
        'pad_token': tokenizer.pad_token,
        'eos_token': tokenizer.eos_token,
        'bos_token': tokenizer.bos_token,
        'unk_token': tokenizer.unk_token,
        'decoder_start_token': tokenizer.pad_token  # Usually same as pad_token
    }
    tokens_path = os.path.join(vocab_dir, f"{sanitized_name}_special_tokens.json")
    with open(tokens_path, 'w', encoding='utf-8') as f:
        json.dump(special_tokens, f, ensure_ascii=False, indent=2)
    
    log_message(f"Vocabulary saved to {vocab_path}")
    log_message(f"Special tokens saved to {tokens_path}")

def process_model_pair(model_pair, output_dir):
    results = {}
    for direction, model_name in model_pair.items():
        log_message(f"Processing direction: {direction}")
        model, tokenizer = download_marianmt_model(model_name)
        tf_model_path = save_tensorflow_model(model, model_name)
        tflite_path = convert_to_tflite(tf_model_path, output_dir, model_name)
        save_vocabulary(tokenizer, output_dir, model_name)
        
        results[direction] = {
            'model_name': model_name,
            'tflite_path': tflite_path
        }
    return results

def verify_conversion(results):
    log_message("Verifying TFLite models and vocab files:")
    for direction, details in results.items():
        log_message(f"Checking {direction} model:")
        tflite_path = details['tflite_path']
        if os.path.exists(tflite_path):
            log_message(f"TFLite model exists: {tflite_path}")
        else:
            log_message(f"TFLite model missing: {tflite_path}")

def main():
    log_message(f"Starting model conversion for language pair: {SELECTED_PAIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = process_model_pair(MODEL_PAIRS[SELECTED_PAIR], OUTPUT_DIR)
    verify_conversion(results)
    log_message("Model conversion process completed successfully!")

if __name__ == "__main__":
    main()