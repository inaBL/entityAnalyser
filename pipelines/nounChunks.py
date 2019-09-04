import spacy
from timeit import default_timer as timer
from datetime import datetime
from sentiment import classSentiment

nlp = spacy.load('en_core_web_lg')
s = classSentiment.Sentiment


# Returns list of noun chunk and root text
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


# Without root word
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


# Returns list of tuples, with list of words consisting being first part of the tuple, and root word the second.
# The first list of words is list of tuples with the word/token of the chunk being paired with the sentiment value
def noun_chunks_sentiment(document) -> list:
    chunk_sentiment = []

    for item in document.noun_chunks:
        chunk = []
        for word in item.text.split():
            chunk.append((word, s.sentiment_single_word(word)))
        chunk_sentiment.append((chunk, item.root.text))

    return chunk_sentiment


if __name__ == '__main__':
    # Set document to be analysed below
    print(f'Starting text to document import at {datetime.now()} ...')
    start = timer()

    with open('/Users/ibl/Documents/entityAnalyser/data/goblin.txt') as f:
        doc = nlp(f.read())

    end = timer()
    print(f'Finished text to document import at {datetime.now()}. '
          f'\nTook {end - start} seconds')

    print(noun_chunks_sentiment(doc))

