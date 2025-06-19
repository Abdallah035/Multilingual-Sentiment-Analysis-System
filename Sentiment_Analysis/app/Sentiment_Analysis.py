import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Hide INFO and WARNING messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from transformers import TFDistilBertForSequenceClassification, AutoTokenizer
from transformers import logging
import tensorflow as tf
logging.set_verbosity_error()

def load_model(model_path):
    try:
        # Get absolute path relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_link = os.path.join(script_dir, model_path)

        if not os.path.isdir(model_link):
            raise FileNotFoundError(f"Model directory not found: {model_link}")

        required_files = ["config.json", "tf_model.h5", "tokenizer.json"]
        missing_files = [file for file in required_files if not os.path.exists(os.path.join(model_link, file))]
        if missing_files:
            raise FileNotFoundError(f"Missing required files in {model_link}: {', '.join(missing_files)}")

        model = TFDistilBertForSequenceClassification.from_pretrained(model_link)
        tokenizer = AutoTokenizer.from_pretrained(model_link)

        print(f"Model and tokenizer loaded successfully from: {model_link}")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None




def predict_sentiment(text,model,tokenizer):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=64)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    predicted_class = probs.argmax()
    return predicted_class, probs


def get_sentiment(text):
    model, tokenizer = load_model("../model/")
    predicted_class, probs = predict_sentiment(text, model, tokenizer)
    if predicted_class == 0:
        return "Negative"
    else:
        return "Positive"


def get_sentiment_list(text_list): #["good" , "bad"]
    res = {"positive": [], "negative": []}
    model , tokenizer = load_model("../model/")
    for i,text in enumerate(text_list):
        predicted , probs = predict_sentiment(text,model,tokenizer)
        if predicted == 1:
           res["positive"].append(i)

        else:
            res["negative"].append(i)

    return res
