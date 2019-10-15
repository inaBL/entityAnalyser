import spacy
import math
from spacy.lemmatizer import Lemmatizer
from timeit import default_timer as timer
from datetime import datetime
from sentiment import classSentiment


# TBI: pandas dataframe compatibility
nlp = spacy.load('en_core_web_lg')
s = classSentiment.Sentiment
lemma = Lemmatizer()


# Returns list of noun chunk and root word text
def noun_chunks_list(document) -> list:
    chunks = [(item.text, item.root.text) for item in document.noun_chunks]
    return chunks


# Returns list of noun chunk and root text, without new line in text
def noun_chunks_list_cleaned(document) -> list:
    chunks = []

    for item in document.noun_chunks:
        if '\n' in item.text:
            chunks.append((item.text.replace('\n', ''), item.root.text))
        else:
            chunks.append((item.text, item.root.text))

    return chunks


# Returns list of noun chunk text, without the root word
def noun_chunks_min(document) -> list:
    chunks = []

    for item in document.noun_chunks:
        if '\n' in item.text:
            chunks.append(item.text.replace('\n', ''))
        else:
            chunks.append(item.text)

    return chunks


# Returns list of tuples with noun chunk split into words and root word
def noun_chunks_words(document) -> list:
    words = []

    for item in document.noun_chunks:
        words.append((item.text.replace('\n', '').split(), item.root.text))

    return words


# Returns list of tuples. Tuple[1] is a list of word sentiment pairs, Tuple[2] is the root word.
# The first list of words is list of tuples with the word/token of the chunk being paired with the sentiment value
# E.g. ([('swift', 1), ('decay', 0)], 'decay')
# NOTE: looks up sentiment value for the lemma for increased coverage
def noun_chunks_sentiment(document) -> list:
    chunk_sentiment = []

    for item in document.noun_chunks:
        chunk = []
        for word in item.text.split():
            chunk.append((word, s.sentiment_single_word(lemma.lookup(word))))
        chunk_sentiment.append((chunk, item.root.text))

    return chunk_sentiment


# Returns list of tuples, Tuple[1] contains the sentiment values of tokens in the noun chunk, Tuple[2] is the root word.
# E.g. ([1, 0], 'decay')
def noun_chunks_roots_sentiment(document) -> list:
    chunk_sentiment = []

    for item in document.noun_chunks:
        chunk = []
        for word in item.text.split():
            chunk.append(s.sentiment_single_word(lemma.lookup(word)))
        chunk_sentiment.append((chunk, item.root.text))

    return chunk_sentiment


# Returns dictionary, with the root word as key and list of the chunk token sentiment values related to the key
# E.g. 'decay': [1, 0]
# Calls function noun_chunks_roots_sentiment()
def noun_chunks_rootword_sentiment(document) -> dict:
    roots_dict = {}
    sentiment, roots = zip(*noun_chunks_roots_sentiment(document))
    l_roots = [token.lower() for token in roots]

    l_sentiment = list(zip(l_roots, sentiment))

    for item in l_sentiment:
        sentiments = item[1]
        if item[0] not in roots_dict.keys():
            roots_dict[item[0]] = sentiments
        elif item[0] in roots_dict.keys():
            sentiments = item[1] + roots_dict[item[0]]
            roots_dict[item[0]] = sentiments

    return roots_dict


def noun_chunks_rootword_sentiment_score(document) -> dict:
    roots_scores = {}

    roots_dict = noun_chunks_rootword_sentiment(document)

    for key in roots_dict.keys():
        if sum(roots_dict[key]) == 0:
            roots_scores[key] = 0
        else:
            roots_scores[key] = round((float((sum(roots_dict[key]) / len(roots_dict[key])))), 4)

    return roots_scores


if __name__ == '__main__':
    # Set document to be analysed below
    print(f'Starting text to document import at {datetime.now()} ...')
    start = timer()

    with open('/Users/ibl/Documents/entityAnalyser/data/goblin.txt') as f:
        doc = nlp(f.read())

    end = timer()
    print(f'Finished text to document import at {datetime.now()}. '
          f'\nTook {end - start} seconds')

    print(noun_chunks_rootword_sentiment_score(doc))

