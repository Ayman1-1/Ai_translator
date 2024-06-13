from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer, GenerationConfig
import os

app = Flask(__name__)

# Define the paths
model_path = './results/model'
tokenizer_path = './results/tokenizer'
generation_config_path = './results/generation_config'

# Check if the paths exist
if not (os.path.isdir(model_path) and os.path.isdir(tokenizer_path) and os.path.isdir(generation_config_path)):
    raise EnvironmentError("Model, tokenizer, or generation config paths are incorrect. Please check the paths.")

# Load the trained model, tokenizer, and generation configuration
try:
    loaded_model = MarianMTModel.from_pretrained(model_path)
    loaded_tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
    loaded_generation_config = GenerationConfig.from_pretrained(generation_config_path)
except Exception as e:
    raise EnvironmentError(f"Error loading model, tokenizer, or generation config: {e}")

def translate(text):
    inputs = loaded_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    generation_kwargs = {
        "max_length": loaded_generation_config.max_length,
        "num_beams": loaded_generation_config.num_beams,
        "bad_words_ids": loaded_generation_config.bad_words_ids,
        "forced_eos_token_id": loaded_generation_config.forced_eos_token_id,
    }
    translated_tokens = loaded_model.generate(**inputs, **generation_kwargs)
    return loaded_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    english_text = request.form['english_text']
    translated_text = translate(english_text)
    return render_template('index.html', english_text=english_text, translated_text=translated_text)

if __name__ == '__main__':
    app.run(debug=True)
