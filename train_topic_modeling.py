import nltk, re
from nltk.tokenize import word_tokenize
from gensim import corpora
import pickle
import gensim
from nltk.corpus import wordnet as wn
import pyLDAvis.gensim
import traceback

# unique stop words
en_stop = set(nltk.corpus.stopwords.words('english'))


def tokenize(text):
    """
    Takes in sentence and split into list of words and returns the same


    :param text: Corpus sentence eg: "things to note"
    :return: list of words eg: ['things', 'to', 'note']
    """

    try:
        words = word_tokenize(text)
        return words
    except:
        return traceback.format_exc()


def get_lemma(word):
    """
    Find a possible base form for the given form, with the given
    part of speech, by checking WordNet's list of exceptional
    forms, and by recursively stripping affixes for this part of
    speech until a form in WordNet is found. -- Reference : wordnet.py

    :param word: tokenized word eg: studies
    :return: lemmatized word eg: study
    """

    try:
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma
    except:
        return traceback.format_exc()


def prepare_text_for_lda(text):
    """
    Cleansing the given sentence which includes removing whitespaces, lowering, removing punctuations,
    tokenizing and Lemmatization

    :param text: corpus sentence
    :return: cleaned list of words
    """

    text = text.strip()
    text = text.lower()
    text = re.sub('[^A-Za-z .-]+', ' ', text)
    text = text.replace('-', '')
    text = text.replace('.', '')
    tokens = tokenize(text)

    if tokens:
        tokens = [token for token in tokens if len(token) > 4]
        tokens = [token for token in tokens if token not in en_stop]
        tokens = [get_lemma(token) for token in tokens]
        return tokens
    else:
        return


if __name__ == '__main__':

    text_data = []

    with open('input.txt', 'r', encoding='utf-8') as f:
        for line in f:
            tokens = prepare_text_for_lda(line)
            if tokens:
                text_data.append(tokens)
            else:
                print("Unable to prepare the data !!")
                break

    """ constructing dictionary of the tokenized words """
    dictionary = corpora.Dictionary(text_data)

    """ building the bag of words, which consist of terms and frequency of occurrence"""
    corpus = [dictionary.doc2bow(text) for text in text_data]

    """ pickling the corpus data """
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')

    """ Number of topics to group together """
    NUM_TOPICS = 5

    """ Fitting in the corpus to LDA model to train the data, 
    To tweak more, try modifying the values of passes"""
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=10)
    ldamodel.save('model10.gensim')

    """ predicted topics result """
    topics = ldamodel.print_topics(num_words=5)
    for topic in topics:
        print(topic)

    """ To visualize the predicted topics, try the same in jupyter """
    # dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    # corpus = pickle.load(open('corpus.pkl', 'rb'))
    # lda = gensim.models.ldamodel.LdaModel.load('model10.gensim')
    #
    # lda10 = gensim.models.ldamodel.LdaModel.load('model10.gensim')
    # lda_display10 = pyLDAvis.gensim.prepare(lda10, corpus, dictionary, sort_topics=False)
    # pyLDAvis.display(lda_display10)
