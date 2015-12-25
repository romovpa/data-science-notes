import pymorphy2
import nltk
import re

class SuperTokenizer(object):
    
    PYMORPH_BAD_GRAMMEMES = ('PNCT', 'PRCL', 'CONJ', 'PREP', 'NPRO', )
    
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.tokenizer = nltk.RegexpTokenizer('\w+|\$[\d\.]+|\S+', flags=re.UNICODE)
        self.stop = nltk.corpus.stopwords.words('russian') + nltk.corpus.stopwords.words('english')
        self.wordnet_lemmatizer = nltk.WordNetLemmatizer()
    
    def normalize_word(self, word):
        lem = self.morph.parse(word)[0]
        if 'LATN' in lem.tag:
            # latin word: use snowball stemmer
            token = self.wordnet_lemmatizer.lemmatize(word)
        elif sum(((g in lem.tag) for g in SuperTokenizer.PYMORPH_BAD_GRAMMEMES)) == 0:
            # good russian word
            token = lem.normal_form
        else:
            token = None
        if token not in self.stop:
            return token
        
    def tokenize_text(self, text):
        text_tokens = re.sub('\W+', ' ', text, flags=re.UNICODE).lower()
        tokens = filter(lambda s: len(s) > 0, text_tokens.split(' '))
        tokens = map(self.normalize_word, tokens)
        tokens = filter(lambda token: token is not None, tokens)
        return tokens
