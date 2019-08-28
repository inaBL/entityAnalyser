import spacy
import string
from timeit import default_timer as timer
from datetime import datetime

nlp = spacy.load('en_core_web_lg')


def noun_chunks_list(document) -> list:
    chunks = []

    for item in document.noun_chunks:
        chunks.append((item.text, item.root.text))

    return chunks


# TBI removal of punctuation containing noun chunks, apart from '
def noun_chunks_list_cleaned(document) -> list:
    chunks = []

    return chunks


if __name__ == '__main__':
    # Set document to be analysed below
    print(f'Starting text to document import at {datetime.now()} ...')
    start = timer()

    with open('/Users/ibl/Documents/entityAnalyser/data/goblin.txt') as f:
        doc = nlp(f.read())

    end = timer()
    print(f'Finished text to document import at {datetime.now()}. '
          f'\nTook {end - start} seconds')

    print(noun_chunks_list_cleaned(noun_chunks_list(doc)))
