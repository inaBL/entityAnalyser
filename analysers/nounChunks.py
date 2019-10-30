'''
Functions for extracting noun-chunk information from text files.
If run directly, writes a csv file at specified location.

En_core_web_lg may take time to load depending on your set up, consider downloading and using sm or md versions as needed.

NOTE: Document to doc conversion takes approx 0,4 ms per token.

INPUT: Text for spacy document object.
OUTPUT: Basic stats in csv file.
'''
import spacy
from spacy.lemmatizer import Lemmatizer
from timeit import default_timer as timer
from datetime import datetime
from sentiment import classSentiment


# TBD: pandas dataframe compatibility

nlp = spacy.load('en_core_web_lg')
nlp.max_length = 1500000
s = classSentiment.Sentiment
lemma = Lemmatizer()


# Returns list of noun chunk and root word text
def nc_list(document) -> list:
    chunks = [(item.text, item.root.text) for item in document.noun_chunks]
    return chunks


# Returns list of noun chunk and root text, without new line in text
def nc_list_cleaned(document) -> list:
    chunks = []

    for item in document.noun_chunks:
        if '\n' in item.text:
            chunks.append((item.text.replace('\n', ''), item.root.text))
        else:
            chunks.append((item.text, item.root.text))

    return chunks


# Returns list of noun chunk text, without the root word
def nc_min(document) -> list:
    chunks = []

    for item in document.noun_chunks:
        if '\n' in item.text:
            chunks.append(item.text.replace('\n', ''))
        else:
            chunks.append(item.text)

    return chunks


# Returns list of tuples with noun chunk split into words and root word
# Tuple[1] being token list and Tuple[2] the rootword
# E.g (['swift', 'decay'], 'decay')
def nc_words(document) -> list:
    words = []

    for item in document.noun_chunks:
        words.append((item.text.replace('\n', '').split(), item.root.text))

    return words


# Returns list of tuples with noun chunk split into lemmas and lemmatised root word in lowercase
def nc_lemmas(document) -> list:
    words = []

    for item in document.noun_chunks:
        chunk = []

        for word in item.text.replace('\n', '').split():
            chunk.append(lemma.lookup(word).lower())

        words.append((chunk, lemma.lookup(item.root.text.lower())))

    return words


# Returns dictionary of all roots as keys + all unique noun chunk tokens related to the root as list
def nc_clusters(document) -> dict:
    clusters = {}

    for item in nc_words(document):
        words = list(set(item[0]))
        if item[1] not in clusters.keys():
            clusters[item[1]] = words
        elif item[1] in clusters.keys():
            words = list(set(item[0] + clusters[item[1]]))
            clusters[item[1]] = words

    return clusters


# Returns dictionary of all unique lemmatised roots as keys + all unique noun chunk lemmas related to the root as list
def nc_lemma_clusters(document) -> dict:
    clusters = {}

    for item in nc_lemmas(document):
        words = list(set(item[0]))
        if item[1] not in clusters.keys():
            clusters[item[1]] = words
        elif item[1] in clusters.keys():
            words = list(set(item[0] + clusters[item[1]]))
            clusters[item[1]] = words

    return clusters


# Returns list of tuples. Tuple[1] is a list of word sentiment pairs, Tuple[2] is the root word.
# The first list of words is list of tuples with the word/token of the chunk being paired with the sentiment value
# E.g. ([('swift', 1), ('decay', 0)], 'decay')
# NOTE: looks up sentiment value for the lemma for increased coverage
def nc_sentiment(document) -> list:
    chunk_sentiment = []

    for item in document.noun_chunks:
        chunk = []
        for word in item.text.split():
            chunk.append((word, s.sentiment_single_word(lemma.lookup(word))))
        chunk_sentiment.append((chunk, item.root.text))

    return chunk_sentiment


# Returns list of tuples, Tuple[1] contains the sentiment values of tokens in the noun chunk, Tuple[2] is the root word.
# E.g. ([1, 0], 'decay')
def nc_roots_sentiment(document) -> list:
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
def nc_root_sentiment(document) -> dict:
    roots_dict = {}
    sentiment, roots = zip(*nc_roots_sentiment(document))
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


def nc_root_sentiment_score(document) -> dict:
    roots_scores = {}

    roots_dict = nc_root_sentiment(document)

    for key in roots_dict.keys():
        if sum(roots_dict[key]) == 0:
            roots_scores[key] = 0
        else:
            roots_scores[key] = round((float((sum(roots_dict[key]) / len(roots_dict[key])))), 4)

    return roots_scores


# Return as default 10 highest sentiment score rootwords, calls nc_root_sentiment_score
def nc_root_mostpos(document, number=10) -> dict:
    return dict(sorted(nc_root_sentiment_score(document).items(),
                       key=lambda x: x[1], reverse=True)[:number])


# Return as default 10 lowest sentiment score rootwords, calls calls nc_root_sentiment_score
def nc_root_mostneg(document, number=10):
    return dict(sorted(nc_root_sentiment_score(document).items(), key=lambda x: x[1])[:number])


# Return list of tokens of noun clusters related to given word
# If the word doesn't appear as root, return empty list
def nc_word_cluster(word: str, document) -> list:
    if word in nc_clusters(document).keys():
        return nc_clusters(document)[word]
    else:
        return []


# Return list of lemmas of noun clusters related to lemmatised given word
# If the word doesn't appear as root, return empty list
def nc_word_lemmas(word: str, document) -> list:
    lemmatised = lemma.lookup(word).lower()
    if lemmatised in nc_lemma_clusters(document).keys():
        return nc_lemma_clusters(document)[lemmatised]
    else:
        return []


if __name__ == '__main__':
    # Set document to be analysed below
    print(f'Starting text to document import at {datetime.now()} ...')
    start = timer()

    with open('/Users/ibl/Documents/entityAnalyser/data/goblin.txt') as f:
        doc = nlp(f.read())

    end = timer()
    print(f'Finished text to document import at {datetime.now()}. '
          f'\nTook {end - start} seconds')

    print(nc_word_lemmas('dish', doc))

