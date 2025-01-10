import os
import tensorflow as tf
from transformers import MarianMTModel, MarianTokenizer, TFAutoModelForSeq2SeqLM

# Configuration
MODELS = {
    'en-fr': "Helsinki-NLP/opus-mt-en-fr",
    'fr-en': "Helsinki-NLP/opus-mt-fr-en",
    'en-de': "Helsinki-NLP/opus-mt-en-de",
    'de-en': "Helsinki-NLP/opus-mt-de-en",
}

# Select the model to convert (change this value to convert different models)
SELECTED_MODEL = 'fr-en'
OUTPUT_DIR = "tflite_models"

def sanitize_model_name(model_name):
    return model_name.replace('/', '_').replace('-', '_')

def download_marianmt_model(model_name):
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def save_tensorflow_model(model, model_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sanitized_name = sanitize_model_name(model_name)
    tf_model_path = os.path.join(base_dir, f"saved_models/{sanitized_name}")
    os.makedirs(os.path.dirname(tf_model_path), exist_ok=True)
    
    # Save the model in TF SavedModel format
    model.save_pretrained(tf_model_path)
    tf.saved_model.save(model, tf_model_path)
    return tf_model_path

def convert_to_tflite(tf_model_path, output_dir, model_name):
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()
        
        os.makedirs(output_dir, exist_ok=True)
        sanitized_name = sanitize_model_name(model_name)
        tflite_model_path = os.path.join(output_dir, f"{sanitized_name}.tflite")
        
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise

def main(model_name, output_dir):
    model, tokenizer = download_marianmt_model(model_name)
    tf_model_path = save_tensorflow_model(model, model_name)
    convert_to_tflite(tf_model_path, output_dir, model_name)

if __name__ == "__main__":
    model_name = MODELS[SELECTED_MODEL]
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_DIR)
    main(model_name, output_dir)