import re
import stanza
import logging
import pandas as pd
from tqdm import tqdm
from typing import List
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
def lemmatize(sent, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    doc = nlp(sent)
    tokens = [word.lemma for sent in doc.sentences for word in sent.words if word.pos in allowed_postags]
    return " ".join(tokens)
    

def generate_collocated_phrases(sentences: List[str]):
    """
    sentences: List[str]
     Lemmatized sentences 
    """
    sents_new = [sent.split(" ") for sent in sentences]
    phrases = Phrases(sents_new, scoring='npmi', min_count=7, threshold=-0.5)

    bigram_model = Phraser(phrases)

    trigram = Phrases(bigram_model[sents_new], scoring='npmi', min_count=7, threshold=-0.5)

    trigram_model = Phraser(trigram)

    return [trigram_model[bigram_model[sent]] for sent in sents_new]


if __name__ == '__main__':

    #*******************SETTING LOGGER ******************** 

    logger = logging.getLogger('transformer-logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler('./logs/sentence-transformer.log')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    #*******************DATA IMPORTS ******************** 

    logger.info("Importing models and references.")

    nlp = stanza.Pipeline(lang='pt', processors='tokenize,mwt,pos,lemma')

    stopwords = read_stopwords("../stopwords/stopwords.txt")

    activities = pd.read_csv("../data/datasets/deputy_activity.csv")

    speech_acts = pd.read_csv("../data/datasets/labelled_data_filtered.csv")

    logger.info("Removing stopwords for activities and speech acts.")

    # Dropping duplicate activities
    unique_activities = activities.drop_duplicates(subset=["time", "name", "key", "text"])

    # Preparing texts for lemmatization
    activity_text_without_stopwords = [remove_stopwords(text, stopwords) for text in unique_activities.loc[:, "text"]]

    speech_acts_without_stopwords = [remove_stopwords(text, stopwords) for text in speech_acts.loc[:, "sentence_text"]]

    # Extracting texts as list
    texts = unique_activities.loc[:, "text"]
    speech_acts_ls = speech_acts.loc[:, "sentence_text"]

    #*******************GENERATING LEMMATIZED SENTENCES ******************** 

    # Lemmatizing activities

    logger.info("Lemmatizing activities…")

    pbar = tqdm(total=len(activity_text_without_stopwords))

    activities_lemmatized = []

    for text in activity_text_without_stopwords:
        text_lemmatized = lemmatize(text)
        activities_lemmatized.append(text_lemmatized)
        pbar.update(1)

    pbar.close()

    # Lemmatizing speech acts

    logger.info("Lemmatizing speech acts…")

    pbar = tqdm(total=len(speech_acts_ls))

    speech_acts_lemmatized = []

    for text in speech_acts_without_stopwords:
        text_lemmatized = lemmatize(text)
        speech_acts_lemmatized.append(text_lemmatized)
        pbar.update(1)

    pbar.close()

    #*******************GENERATING PHRASE COLLOCATIONS ******************** 

    logger.info("Generating collocations for speech acts and activities.")

    collocated_phrases_speech_acts = generate_collocated_phrases(speech_acts_lemmatized)

    collocated_phrases_activities = generate_collocated_phrases(activities_lemmatized)

    #*******************GENERATING SENTENCE-TRANSFORMER EMBEDDINGS ******************** 

    logger.info("Retrieving sentence-transformers…")

    # Portuguese Sentence Transformer (PT_MODEL)
    pt_model = SentenceTransformer("ricardo-filho/bert-portuguese-cased-nli-assin-assin-2")

    # Multilingual Sentence Transformer (ML_MODEL)
    ml_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

    # Neural Mind Transformer (NM_MODEL)
    nm_model = SentenceTransformer("neuralmind/bert-base-portuguese-cased")

    # Encoding activities using PT_MODEL

    logger.info("Encoding activities…")

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


    # Encoding activities using NM_MODEL
    pbar = tqdm(total=len(texts))

    activities_encoded_nm = []

    for sent in texts:
        embedding = nm_model.encode(sent)
        activities_encoded_nm.append(embedding)
        pbar.update(1)

    pbar.close()

    # Encoding speech acts using PT_MODEL

    logger.info("Encoding speech acts…")

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


    # Encoding speech acts using NM_MODEL
    pbar = tqdm(total=len(speech_acts_ls))

    speech_acts_encoded_nm = []

    for speech_act in speech_acts_ls:
        embedding = nm_model.encode(speech_act)
        speech_acts_encoded_nm.append(embedding)
        pbar.update(1)

    pbar.close()

    #*******************CREATING DATAFRAMES FOR EXPORT ******************** 

    logger.info("Preparing encoded DataFrames for export…")

    unique_activities.loc[:, "encoded_activities_ml"] = activities_encoded_ml
    speech_acts.loc[:, "encoded_speech_acts_ml"] = speech_acts_encoded_ml

    unique_activities.loc[:, "encoded_activities_pt"] = activities_encoded_pt
    speech_acts.loc[:, "encoded_speech_acts_pt"] = speech_acts_encoded_pt

    unique_activities.loc[:, "encoded_activities_nm"] = activities_encoded_nm
    speech_acts.loc[:, "encoded_speech_acts_nm"] = speech_acts_encoded_nm

    unique_activities.loc[:, "activity_phrases"] = collocated_phrases_activities
    speech_acts.loc[:, "speech_act_phrases"] = collocated_phrases_speech_acts

    #*******************CREATING DATAFRAMES FOR EXPORT ******************** 

    unique_activities.to_csv("../data/encoded_datasets/activities_encoded.csv", index=False)
    speech_acts.to_csv("../data/encoded_datasets/speech_acts_encoded.csv", index=False)

    logger.info("DataFrames exported. Wrapping up…")