import os
import json
import numpy as np
import tensorflow as tf

SPIECE_UNDERLINE = "▁"

class MarianTranslator:
    def __init__(self, model_dir, lang_pair):
        self.model_dir = model_dir
        self.lang_pair = lang_pair
        self.models = {}
        self.vocab = {}
        self.special_tokens = {}

        # Load models and vocabs
        for direction in ['forward', 'backward']:
            model_name = f"Helsinki_NLP_opus_mt_{lang_pair.replace('-', '_')}"
            if direction == 'backward':
                model_name = f"Helsinki_NLP_opus_mt_{lang_pair.split('-')[1]}_{lang_pair.split('-')[0]}"
            
            # Load TFLite model
            model_path = os.path.join(model_dir, f"{model_name}.tflite")
            self.models[direction] = tf.lite.Interpreter(model_path=model_path)
            self.models[direction].allocate_tensors()
            
            # Load vocabulary
            vocab_path = os.path.join(model_dir, 'vocab', f"{model_name}_vocab.json")
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
                self.vocab[direction] = vocab_data['vocab']
            
            # Load special tokens
            tokens_path = os.path.join(model_dir, 'vocab', f"{model_name}_special_tokens.json")
            with open(tokens_path, 'r', encoding='utf-8') as f:
                self.special_tokens[direction] = json.load(f)

    def tokenize(self, text, direction):
        vocab = self.vocab[direction]
        special_tokens = self.special_tokens[direction]
        
        # Normalize text and split into words
        words = text.lower().strip().split()
        token_ids = []
        
        # Add start token
        if special_tokens['bos_token'] in vocab:
            token_ids.append(vocab[special_tokens['bos_token']])
        
        # Tokenize each word
        for word in words:
            # Add word with SPIECE_UNDERLINE prefix
            token = SPIECE_UNDERLINE + word
            if token in vocab:
                token_ids.append(vocab[token])
            else:
                # Handle unknown tokens
                for char in word:
                    char_token = SPIECE_UNDERLINE + char
                    if char_token in vocab:
                        token_ids.append(vocab[char_token])
                    else:
                        token_ids.append(vocab[special_tokens['unk_token']])
        
        # Add end token
        if special_tokens['eos_token'] in vocab:
            token_ids.append(vocab[special_tokens['eos_token']])
            
        return token_ids

    def detokenize(self, token_ids, direction):
        vocab_inv = {v: k for k, v in self.vocab[direction].items()}
        special_tokens = set(self.special_tokens[direction].values())
        
        tokens = []
        for idx in token_ids:
            if idx in vocab_inv:
                token = vocab_inv[idx]
                # Skip special tokens and empty tokens
                if token not in special_tokens and token != '':
                    tokens.append(token)
        
        # Join and clean up the tokens
        text = ''.join(tokens)
        # Replace SPIECE_UNDERLINE with space and handle multiple spaces
        text = text.replace(SPIECE_UNDERLINE, ' ').strip()
        text = ' '.join(text.split())
        return text

    def translate(self, text, direction='forward'):
        interpreter = self.models[direction]
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Tokenize and prepare input
        token_ids = self.tokenize(text, direction)
        seq_len = len(token_ids)
        
        # Prepare all input tensors
        inputs = {
            'input_ids': np.array([token_ids], dtype=np.int32),
            'attention_mask': np.ones((1, seq_len), dtype=np.int32),
            'decoder_input_ids': np.array([[0]], dtype=np.int32),  # Start token
            'decoder_attention_mask': np.ones((1, 1), dtype=np.int32)
        }
        
        # Set input tensors
        for detail in input_details:
            input_name = detail['name'].split(':')[0].split('_')[-2:]
            input_name = '_'.join(input_name)
            if input_name in inputs:
                tensor_data = inputs[input_name]
                interpreter.resize_tensor_input(detail['index'], tensor_data.shape)
        
        interpreter.allocate_tensors()
        
        # Set the tensors after allocation
        for detail in input_details:
            input_name = detail['name'].split(':')[0].split('_')[-2:]
            input_name = '_'.join(input_name)
            if input_name in inputs:
                interpreter.set_tensor(detail['index'], inputs[input_name])
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor with logits
        output_data = None
        for detail in output_details:
            if len(detail['shape']) == 3:  # Looking for [batch_size, seq_len, vocab_size]
                output_data = interpreter.get_tensor(detail['index'])[0]
                break
        
        if output_data is None:
            raise ValueError("Could not find appropriate output tensor")
        
        # Get predicted token ids
        predicted_ids = np.argmax(output_data, axis=-1)
        return self.detokenize(predicted_ids, direction)

def inspect_model(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")

def main():
    model_dir = "tflite_models"
    lang_pair = "en-fr"
    translator = MarianTranslator(model_dir, lang_pair)
    
    # Inspect models
    for direction, interpreter in translator.models.items():
        print(f"\nInspecting {direction} model:")
        inspect_model(interpreter)
    
    # Test translations
    test_sentences = ["Hello, how are you?", "I love programming."]
    for sentence in test_sentences:
        translation = translator.translate(sentence, direction='forward')
        print(f"Input: {sentence}")
        print(f"Translation: {translation}")

if __name__ == "__main__":
    main()