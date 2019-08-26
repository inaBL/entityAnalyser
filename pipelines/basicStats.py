'''
Functions for basic stat gathering from text files.
If run directly, writes a csv file with the stats at specified location.
En_core_web_lg may take time to load depending on your set up, consider downloading and using sm or md versions as needed.

NOTE: Document to doc conversion takes approx 0,4 ms per token.

INPUT: Text for spacy document object.
OUTPUT: Basic stats in csv file.
'''

import spacy
import csv
from timeit import default_timer as timer
from datetime import datetime
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter

nlp = spacy.load('en_core_web_lg')


def get_token_count(document) -> list:
    tokens = [('Total tokens', len(document))]

    return tokens


# Return as default 10 most common tokens and their counts.
def get_top_tokens(document, number=10) -> list:
    tokens = [token.text for token in document if token.is_punct is not True]
    return Counter(tokens).most_common(number)


# Return as default 10 most common tokens and their counts , w/o punct or stop words, for full list print STOP_WORDS
# STOP_WORDS imported as workaround, en_core_web_lg has known bug with stop words.
def get_top_tokens_cleaned(document, number=10) -> list:
    tokens = [token.text for token in document if '\n' not in token.text
              and ' ' not in token.text
              and token.is_punct is not True
              and token.text not in STOP_WORDS]

    return Counter(tokens).most_common(number)


# Return as default 10 most common named entities and their counts
def get_named_entities(document, number=10) -> list:
    entities = [ent.text for ent in document.ents if '\n' not in ent.text]

    return Counter(entities).most_common(number)


# Return as default 10 most common lemmas not including punctuation
def get_top_lemmas(document, number=10) -> list:
    lemmas = [token.lemma_ for token in document if token.is_punct is not True]

    return Counter(lemmas).most_common(number)


# Return as default 10 most common lemmas not including punctuation, new lines, or stop words
def get_top_lemmas_cleaned(document, number=10) -> list:
    lemmas = [token.lemma_ for token in document if '\n' not in token.text
              and ' ' not in token.text
              and token.is_punct is not True
              and token.text not in STOP_WORDS]

    return Counter(lemmas).most_common(number)


# Return all POS-tags and their frequencies in descending order
def get_pos(document) -> list:
    pos = [token.pos_ for token in document]

    return Counter(pos).most_common()


if __name__ == '__main__':
    # Set document to be analysed below
    print(f'Starting text to document import at {datetime.now()} ...')
    start = timer()

    with open('/Users/ibl/Documents/entityAnalyser/data/dracula.txt') as f:
        doc = nlp(f.read())

    end = timer()
    print(f'Finished text to document import at {datetime.now()}. '
          f'\nTook {end - start} seconds')

    # Set preferred stat file location below, if no file exists, new one will be created.
    # NOTE: Writes over file
    with open('/Users/ibl/Documents/entityAnalyser/stats/dracula_stats.csv', 'w+', newline='\n') as csvfile:
        statwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for item in get_token_count(doc):
            statwriter.writerow(item)
        statwriter.writerow("")
        statwriter.writerow(["Top tokens"])
        for item in get_top_tokens_cleaned(doc):
            statwriter.writerow(item)
        statwriter.writerow("")
        statwriter.writerow(["Top lemmas"])
        for item in get_top_lemmas_cleaned(doc):
            statwriter.writerow(item)
        statwriter.writerow("")
        statwriter.writerow(["Named entities"])
        for item in get_named_entities(doc):
            statwriter.writerow(item)
        statwriter.writerow("")
        statwriter.writerow(["POS-tags"])
        for item in get_pos(doc):
            statwriter.writerow(item)

    print("Finished creating the file")
