from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from gensim import models, corpora
import hashlib
import csv, traceback


# Load Saved LDA Model
ldamodel = models.LdaModel.load('model10.gensim')

topics = ldamodel.print_topics(num_words=5)

""" Modify the value as per the corresponding to the predicted/trained results"""
labels = {0: "software_access", 1: "user_access", 2: "provide_access", 3: "software_access_error", 4: "admin_access",
          5: "Business_report"}

STOPWORDS = stopwords.words('english')

# load corpora dictionary
loaded_dictionary = corpora.Dictionary.load('dictionary.gensim')


def clean_text(text):
    """
    Tokenize the word and remove stop words

    :param text: user questions
    :return: cleansed list of words
    """

    try:
        tokenized_text = word_tokenize(text.lower())
        cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
        return cleaned_text
    except:
        return traceback.format_exc()


def get_hashed_text(text):
    """
    Hashes the given sentence

    :param text: user sentence
    :return: hexa value
    """

    try:
        hashed_value = hashlib.sha512(text.encode('utf-8')).hexdigest()
        return hashed_value
    except:
        return traceback.format_exc()


def get_unique_quests(file_name):
    """
    Finds the unique sentences for the given file

    :param file_name: input file name
    :return: list of unique questions for the given file

    """
    unique_quests = {}
    unique_quests_list = []

    with open(file_name, 'r', encoding='utf-8') as f:
        for text in f:
            cleaned_text = clean_text(text)
            input_text = ' '.join(cleaned_text)
            hashed_text = get_hashed_text(str(input_text))
            if unique_quests.get(hashed_text):
                unique_quests[hashed_text] += 1
            else:
                unique_quests[hashed_text] = 1
                unique_quests_list.append(input_text)

    return unique_quests_list


def map_categorized_data():
    """
    Takes in test data and finds the labels from the trained data

    :return: dictionary of labels and corresponding grouped sentences
    """
    categorized_data = {}
    quests = get_unique_quests("input.txt")
    if quests:
        for text in quests:
            bow = loaded_dictionary.doc2bow(clean_text(text))
            label =labels.get(max(dict(ldamodel[bow]), key=dict(ldamodel[bow]).get))
            if categorized_data.get(label):
                existing_quest = categorized_data.get(label)
                existing_quest.append(text)
                categorized_data[label] = existing_quest
            else:

                categorized_data[label] = [text]

    return categorized_data


def write_to_csv(output_file):
    """
    Writes the grouped category to csv file
    
    :param output_file: file name of the data to be written
    """
    d = map_categorized_data()
    if d:
        keys = sorted(d.keys())
        with open(output_file, "w", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(keys)
            writer.writerows(zip(*[d[str(key)] for key in keys]))

    else:
        print("No data found !!")


if __name__ == '__main__':
    write_to_csv("tkts_training_data_test.csv")

