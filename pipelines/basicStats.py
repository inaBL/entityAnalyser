import spacy
from collections import Counter

nlp = spacy.load('en_core_web_lg')
file_path = ""


def get_token_count(document) -> int:
    return len(document)


def get_top_tokens(document, number=10) -> list:
    tokens = [token.text for token in document if token.is_punct is not True]
    return Counter(tokens).most_common(number)


# Issue with Spacy stopwords not recognising members, usecase 'and'. need to debug
def get_top_tokens_cleaned(document, number=10) -> list:
    tokens = [token.text for token in document if token.is_punct is not True
              and token.is_stop is not True]

    return Counter(tokens).most_common(number)


if __name__ == '__main__':
    pass
    # Set document to be analysed below
    # with open('/Users/ibl/Documents/nlpProject/goblin.txt') as f:
        # doc = nlp(f.read())
