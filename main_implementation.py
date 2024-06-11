import pandas as pd
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoModelForSequenceClassification

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load models and tokenizers
tokenizer_summarization = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
model_summarization = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")

tokenizer_sentiment = AutoTokenizer.from_pretrained("saved_model")
model_sentiment = AutoModelForSequenceClassification.from_pretrained("saved_model")

ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")

# Initialize pipelines
summarization_pipeline = pipeline("summarization", model=model_summarization, tokenizer=tokenizer_summarization)
sentiment_pipeline = pipeline("sentiment-analysis", model=model_sentiment, tokenizer=tokenizer_sentiment)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)


# Helper functions
def summarize_text(text, max_length=150, num_return_sequences=1):
    inputs = tokenizer_summarization.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model_summarization.generate(
        inputs, max_length=max_length, num_return_sequences=num_return_sequences, num_beams=4, early_stopping=True
    )
    summary = tokenizer_summarization.decode(outputs[0], skip_special_tokens=True)
    return summary


def analyze_sentiment(text):
    if len(text) > tokenizer_sentiment.model_max_length:
        text = text[:tokenizer_sentiment.model_max_length]  # Truncate text
    return sentiment_pipeline(text)


def recognize_entities(text):
    return ner_pipeline(text)


# Main processing function
def main():
    df = pd.read_csv('legal_text - Sheet1.csv')  # Update with your file path
    df_subset = df[['case_text']].dropna()

    # Process each document
    for idx, row in df_subset.iterrows():
        text = row['case_text']

        # Summarize text
        try:
            summary = summarize_text(text)
        except Exception as e:
            summary = "Error in summarization: " + str(e)

        # Sentiment analysis
        try:
            sentiment_results = analyze_sentiment(text)
        except Exception as e:
            sentiment_results = "Error in sentiment analysis: " + str(e)

        # Named entity recognition
        try:
            ner_results = recognize_entities(text)
        except Exception as e:
            ner_results = "Error in NER: " + str(e)

        print(f"Document {idx + 1} Summary:\n", summary)
        print(f"Document {idx + 1} Sentiment Analysis:\n", sentiment_results)
        print(f"Document {idx + 1} Named Entity Recognition:\n", ner_results)


if __name__ == "__main__":
    main()
