'''
Functions for basic stat gathering from text files. If run directly, writes a csv file with the stats.

INPUT: Text-file, .txt assumed, for pdf import and use pandas.
OUTPUT: Basic stats in csv file.
'''

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter

nlp = spacy.load('en_core_web_lg')


def get_token_count(document) -> int:
    return len(document)


# Return as default 10 most common tokens and their counts.
def get_top_tokens(document, number=10) -> list:
    tokens = [token.text for token in document if token.is_punct is not True]
    return Counter(tokens).most_common(number)


# Return as default 10 most common tokens and their counts , w/o punct or stop words, for full list print STOP_WORDS
# STOP_WORDS imported as workaround, en_core_web_lg has known bug with stop words.
def get_top_tokens_cleaned(document, number=10) -> list:
    tokens = [token.text for token in document if '\n' not in token.text
              and token.is_punct is not True
              and token.text not in STOP_WORDS]

    return Counter(tokens).most_common(number)


if __name__ == '__main__':
    # pass
    # Set document to be analysed below
    with open('/Users/ibl/Documents/entityAnalyser/data/goblin.txt') as f:
        doc = nlp(f.read())

    print(get_top_tokens_cleaned(doc))
