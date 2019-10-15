from sentiment.classNegative import *
from sentiment.classPositive import *
import regex as re
from timeit import default_timer as timer
from datetime import datetime


class Sentiment(Positive, Negative):
    def __init__(self):
        super().__init__()

    @staticmethod
    def sentiment_text(text_list: list) -> list:
        sentiment = []

        for word in text_list:
            if word is not re.search(r"\p{P}+", word):
                if Sentiment().is_positive(word):
                    sentiment.append((word, 1))
                elif Sentiment().is_negative(word):
                    sentiment.append((word, -1))
                else:
                    sentiment.append((word, 0))
            else:
                pass

        return sentiment

    @staticmethod
    def sentiment_text_values(text_list: list) -> list:
        sentiment = []

        for word in text_list:
            if word is not re.search(r"\p{P}+", word):
                if Sentiment().is_positive(word):
                    sentiment.append(1)
                elif Sentiment().is_negative(word):
                    sentiment.append(-1)
                else:
                    sentiment.append(0)
            else:
                pass

        return sentiment

    @staticmethod
    def sentiment_single_word(word: str) -> int:
        if word is re.search(r"\p{P}+", word):
            return 0
        elif Sentiment().is_positive(word):
            return 1
        elif Sentiment().is_negative(word):
            return -1
        else:
            return 0

    @staticmethod
    def sentiment_score_text(text) -> float:
        score = [item[1] for item in Sentiment().sentiment_text(text)]
        return float(sum(score) / len(score))

    # TBI: stop word removal for sentiment score definition (lessen noise, amplify emotional features)


if __name__ == '__main__':
    text = " I, a princess, king-descended, decked with jewels, gilded, drest, " \
           "Would rather be a peasant with her baby at her breast, " \
           "For all I shine so like the sun, and am purple like the west." \
           "Two and two my guards behind, two and two before," \
           "Two and two on either hand, they guard me evermore;" \
           "Me, poor dove, that must not coo--eagle that must not soar." \
           "All my fountains cast up perfumes, all my gardens grow " \
           "Scented woods and foreign spices, with all flowers in blow " \
           "That are costly, out of season as the seasons go.".split()

    print(f'Starting sentiment calculation at {datetime.now()} ...')
    start = timer()

    print(Sentiment.sentiment_text(text))
    print(Sentiment.sentiment_score_text(text))

    end = timer()
    print(f'Finished sentiment calculation at  {datetime.now()}. '
          f'\nTook {end - start} seconds')
