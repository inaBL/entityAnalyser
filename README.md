# Entity Analyser
This is a personal project for analysing named entity treatment in texts, based on conceptual blending theory of semantic processing.

The pipelines are developed for purpise faster automated processing of named entities and noun chunks, with assumption of the noun chunk tokens and other dependents of the head being indicative of emotional value of the entity in question.

The goal is to obtain the sentiment and topics attached to the entities, and enable data export to further platforms.

### Technical
Used packages will be included in virtual environment. 
Main components listed below.

* Python 3.7
* Spacy 2.0.16
* PyCharm 2019.2.2

### About Sentiment 

Sentiment classes use the opinion lexicons by Hu and Liu. Please see the original publication for more information.

Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews.", Proceedings of the ACM SIGKDD International Conference on Knowledge, Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, Washington, USA

