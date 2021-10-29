import spacy
import pandas as pd
from tqdm import tqdm
from typing import *
from gensim.models.phrases import Phrases, Phraser
from sentence_transformers import SentenceTransformer


def read_stopwords(path: str) -> List[str]:
    with open(path) as f:
        lines = f.readlines()
        stopwords = [line.strip() for line in lines]
    return stopwords


def remove_stopwords(sent: str, stopwords: List[str]) -> str:
    """ Removes stopwords from a given sentence"""
    tokens = re.split(r"\s+", sent.lower())
    tokens_without_stopwords = [token for token in tokens if token not in stopwords]
    clean_sent = ' '.join(tokens_without_stopwords)
    return clean_sent


# Insert lemmatization step
def lemmatize(sent, allowed_postags=['NOUN', 'ADJ']):#, 'VERB', 'ADV']):
    doc = nlp(sent)
    tokens = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
    return " ".join(tokens)

def generate_collocated_phrases(sentences: List[str]):
    """
    sentences: List[str]
     Lemmatized sentences 
    """
    sents_new = [sent.split(" ") for sent in sentences]
    phrases = Phrases(sents_new, min_count=5, threshold=20)

    bigram_model = Phraser(phrases)

    trigram = Phrases(bigram_model[sents_new], min_count=5, threshold=20)

    trigram_model = Phraser(trigram)

    return [trigram_model[bigram_model[sent]] for sent in sents_new]


if __name__ == '__main__':

    nlp = spacy.load('pt_core_news_lg')

    stopwords = read_stopwords("../stopwords/stopwords.txt")

    activities = pd.read_csv("../data/datasets/deputy_activity.csv")

    speech_acts = pd.read_csv("../data/datasets/labelled_data_filtered.csv")

    # Preparing texts for lemmatization
    activity_text_without_stopwords = [remove_stopwords(text, stopwords) for text in activities.loc[:, "text"]]

    speech_acts_without_stopwords = [remove_stopwords(text, stopwords) for text in speech_acts.loc[:, "sentence_text"]]

    # Dropping duplicate activities
    unique_activities = activities.drop_duplicates(subset=["time", "name", "key"])

    # Extracting texts as list
    texts = unique_activities.loc[:, "text"]
    speech_acts_ls = speech_acts.loc[:, "sentence_text"]

    #*******************GENERATING LEMMATIZED SENTENCES ******************** 

    # Lemmatizing activities

    pbar = tqdm(total=len(activity_text_without_stopwords))

    activities_lemmatized = []

    for text in activity_text_without_stopwords:
        text_lemmatized = lemmatize(text)
        activities_lemmatized.append(text_lemmatized)
        pbar.update(1)

    pbar.close()

    # Lemmatizing speech acts

    pbar = tqdm(total=len(speech_acts_ls))

    speech_acts_lemmatized = []

    for text in speech_acts_ls:
        text_lemmatized = lemmatize(text)
        speech_acts_lemmatized.append(text_lemmatized)
        pbar.update(1)

    pbar.close()

    #*******************GENERATING PHRASE COLLOCATIONS ******************** 

    collacted_phrases_speech_acts = generate_collocated_phrases(speech_acts_ls)

    collocated_phrases_activities = generate_collocated_phrases(texts)

    #*******************GENERATING SENTENCE-TRANSFORMER EMBEDDINGS ******************** 

    # Portuguese Sentence Transformer (PT_MODEL)
    pt_model = SentenceTransformer("ricardo-filho/bert-portuguese-cased-nli-assin-assin-2")

    # Multilingual Sentence Transformer (ML_MODEL)
    ml_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

    # Encoding activities using PT_MODEL
    pbar = tqdm(total=len(texts))

    activities_encoded_pt = []

    for sent in texts:
        embedding = pt_model.encode(sent)
        activities_encoded_pt.append(embedding)
        pbar.update(1)

    pbar.close()

    # Encoding activities using ML_MODEL
    pbar = tqdm(total=len(texts))

    activities_encoded_ml = []

    for sent in texts:
        embedding = ml_model.encode(sent)
        activities_encoded_ml.append(embedding)
        pbar.update(1)

    pbar.close()

    # Encoding speech acts using PT_MODEL
    pbar = tqdm(total=len(speech_acts_ls))

    speech_acts_encoded_pt = []

    for speech_act in speech_acts_ls:
        embedding = pt_model.encode(speech_act)
        speech_acts_encoded_pt.append(embedding)
        pbar.update(1)

    pbar.close()

    
    # Encoding speech acts using ML_MODEL
    pbar = tqdm(total=len(speech_acts_ls))

    speech_acts_encoded_ml = []

    for speech_act in speech_acts_ls:
        embedding = ml_model.encode(speech_act)
        speech_acts_encoded_ml.append(embedding)
        pbar.update(1)

    pbar.close()

    #*******************CREATING DATAFRAMES FOR EXPORT ******************** 

    unique_activities.loc[:, "encoded_activities_ml"] = activities_encoded_ml
    speech_acts.loc[:, "encoded_speech_acts_ml"] = speech_acts_encoded_ml

    unique_activities.loc[:, "encoded_activities_pt"] = activities_encoded_pt
    speech_acts.loc[:, "encoded_speech_act"] = speech_acts_encoded_pt

    unique_activities.loc[:, "activity_phrases"] = collocated_phrases_activities
    speech_acts.loc[:, "speech_act_phrases"] = collacted_phrases_speech_acts

    #*******************CREATING DATAFRAMES FOR EXPORT ******************** 

    unique_activities.to_csv("../data/encoded_datasets/activities_encoded.csv", index=False)
    speech_acts.to_csv("../data/encoded_datasets/speech_acts_encoded.csv", index=False)