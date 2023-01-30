import pandas as pd
import nltk
import pytextrank
import spacy
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.tokenize import sent_tokenize
from pytextrank import parse_doc, pretty_print
# from nltk import TextRank

nltk.download('vader_lexicon')
nltk.download('punkt')

def process_chat_data(csv_file: str):
    # Test that the file passed as an argument exists and is a file
    test_process_chat_data(csv_file)
    # Read the CSV file into a DataFrame and parse the Message Date column as a datetime object
    df = pd.read_csv(csv_file, parse_dates=['Message Date','Delivered Date','Read Date','Edited Date'])
    # Convert the Message Date column to a datetime object
    df['conversation_id'] = df['Service'] + '_' + df['Type'] + '_' + df['Message Date'].dt.date.astype(str)


    # Perform the necessary processing on the data
    conversations = df.groupby('conversation_id')

    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()

    for conversation_id, messages in conversations:
        text = messages['Text'].str.cat(sep=' ')
        sentiment = sia.polarity_scores(text)
        keywords = TextBlob(text).noun_phrases
        print(f'Conversation ID: {conversation_id}')
        print(f'Sentiment: {sentiment}, keywords: {keywords}')

        # Sumy LsaSummarizer
        summarizer = LsaSummarizer(Stemmer('english'))
        summarizer.stop_words = get_stop_words('english')
        summary = ' '.join(map(str, summarizer(Tokenizer('english').to_words(text), 5)))
        print(f'Summary (Sumy LsaSummarizer): {summary}')

        # PyTextRank
        nlp = spacy.load("en_core_web_sm")
        tr = TextRank()
        nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
        doc = nlp(text)
        summary = ' '.join([sent.text for sent in doc._.textrank.summary(normalize=True)])
        print(f'Summary (PyTextRank): {summary}')

        # Spacy TextRank
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        summary = ' '.join([sent.text for sent in doc._.textrank_summary(normalize=True)])
        print(f'Summary (Spacy TextRank): {summary}')

        # NLTK TextRank
        summary = summarize(text, ratio=0.2)
        print(f'Summary (NLTK TextRank): {summary}')


import os

def test_process_chat_data(csv_file: str):
    # Test that the file passed as an argument exists and is a file
    assert os.path.isfile(csv_file) == True

    # Test that the file passed as an argument is a CSV file
    assert csv_file.endswith('.csv') == True
