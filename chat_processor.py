import pandas as pd
import nltk
import spacy
import pytextrank
from textblob import TextBlob
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.models.dom import ObjectDocumentModel
from pprint import pprint

# nltk.download('vader_lexicon')
# nltk.download('punkt')

def process_chat_data(csv_file: str):
    LANGUAGE = "english"
    SENTENCES_COUNT = 3
    # Test that the file passed as an argument exists and is a file
    test_process_chat_data(csv_file)
    # Read the CSV file into a DataFrame and parse the Message Date column as a datetime object
    df = pd.read_csv(csv_file, parse_dates=['Message Date','Delivered Date','Read Date','Edited Date'])
    # Convert the Message Date column to a datetime object
    df['conversation_id'] = df['Service'] + '_' + df['Message Date'].dt.date.astype(str)


    # Perform the necessary processing on the data
    conversations = df.groupby('conversation_id')

    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    stemmer = Stemmer('english')

    for conversation_id, messages in conversations:
        text = messages['Text'].str.cat(sep=' ')
        sentiment = sia.polarity_scores(text)
        keywords = TextBlob(text).noun_phrases
        print(f'Conversation ID: {conversation_id}')
        print(f'Sentiment: {sentiment}, keywords: {keywords}')

        # Tokenize text
        parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))

        # Sumy LsaSummarizer
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words('english')
        summary = ' '.join(map(str, summarizer(parser.document, SENTENCES_COUNT)))
        print(f'Summary (Sumy LsaSummarizer): {summary}')

        # Spacy TextRank
        nlp = spacy.load("en_core_web_lg")
        # nlp = spacy.load("en_core_web_sm")
        summary = extract_sentences(text, nlp, 3)
        print(f'Summary (Spacy TextRank): {summary}')

        # PyTextRank
        nlp2 = spacy.load("en_core_web_lg")
        # add PyTextRank to the spaCy pipeline
        nlp2.add_pipe("textrank")
        # examine the top-ranked phrases in the document        
        summary = extract_sentences(text, nlp2, 3)
        print(f'Summary (PyTextRank): {summary}')

        # NLTK TextRank
        parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, 5)
        print(f'Summary (NLTK Sumy): {summary}')


import os

def test_process_chat_data(csv_file: str):
    # Test that the file passed as an argument exists and is a file
    assert os.path.isfile(csv_file) == True

    # Test that the file passed as an argument is a CSV file
    assert csv_file.endswith('.csv') == True

from math import sqrt

def extract_sentences(text, model, number):
    if text:
        # get important sentences from description with the help of spaCy
        doc = model(text)
        # Doc.set_extension('phrases', default=True, force=True)
        sent_bounds = [[sent.start, sent.end, set([])] for sent in doc.sents]
        limit_phrases = 4
        phrase_id = 0
        unit_vector = []
        # add phrases
        phrases = []
        # Check if ._.phrases exists and only overwrite when len() == 0
        if not doc.has_extension('phrases') or (
            doc.has_extension('phrases') and len(doc._.phrases) == 0
        ):
            doc.set_extension('phrases', default={}, force=True)
            doc._.phrases = phrases
        # examine top-ranked phrases
        for phrase in doc._.phrases:
            unit_vector.append(phrase.rank)
            for chunk in phrase.chunks:
                # construct a list of the sentence boundaries
                for sent_start, sent_end, sent_vector in sent_bounds:
                    # iterate through the top-ranked phrases, add them to the vector for each sentence
                    if chunk.start >= sent_start and chunk.end <= sent_end:
                        sent_vector.add(phrase_id)
                        break
            phrase_id += 1
            if phrase_id >= limit_phrases:
                break
        sum_ranks = sum(unit_vector)
        # normalize the vector
        unit_vector = [rank / sum_ranks for rank in unit_vector]
        sent_rank = []
        sent_id = 0
        # iterate through the sentences, add their euclidean distance from the unit vector
        for sent_start, sent_end, sent_vector in sent_bounds:
            sum_sq = 0.0
            for phrase_id in range(len(unit_vector)):
                if phrase_id not in sent_vector:
                    sum_sq += unit_vector[phrase_id] ** 2
            sent_rank.append({'sent_id': sent_id, 'rank': sqrt(sum_sq), 'sentence': doc[sent_start:sent_end].text})
            sent_id += 1
        # sort the sentences by their rank
        sent_rank = sorted(sent_rank, key=lambda x: x['rank'])
        # extract the n most important sentences (with the lowest euclidean distance)
        shortened_description = ''
        for i in range(min(number, len(sent_rank))):
            if shortened_description == '':
                shortened_description = sent_rank[i]['sentence']
            else:
                shortened_description += ' ' + sent_rank[i]['sentence']
        # shortened_description = sent_rank[0]['sentence'] + ' ' + sent_rank[1]['sentence']
        return shortened_description