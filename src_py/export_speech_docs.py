import click
import os
import re
import glob
import toml
import ujson
import uuid
import stanza

import pandas as pd
import numpy as np

from typing import List, Tuple 
from functools import reduce
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


def load_config():
  # load toml file to dictionary
  configs = toml.load(open('./config/config.toml'))
  return configs


def load_speeches(data_dir: str) -> pd.DataFrame:
  most_recent_file = max(glob.iglob(f'{data_dir}/*.parquet'),key=os.path.getctime)
  return pd.read_parquet(most_recent_file)


def speech_filter(speech: str):
  
  split_speech = speech.split(' ')
  
  if len(split_speech) > 5:
    return speech
  else:
    return np.nan
  

def filter_speeches(speeches: pd.DataFrame):
  speeches.loc[:, 'filtered_text'] = speeches.loc[:, 'text'].apply(speech_filter)
  return speeches.dropna(subset=['filtered_text'])
    

def make_date_string(timestamp):
  timestamp = str(timestamp)

  date_regex = re.compile(r'^\d{2,}\-\d{2,}\-\d{2,}')

  return re.search(date_regex, timestamp).group(0)


def create_actor_discourse_string(speeches: pd.DataFrame) -> List[List[dict]]:
  speeches.loc[:, 'time'] = speeches.loc[:, 'time'].apply(make_date_string)
  transformed_text = speeches.groupby(['name', 'time'])['text'].apply(lambda x: ' '.join(x))
  discourse_strings = list(zip(transformed_text.values, transformed_text.index))
  return discourse_strings


def generate_document_json(discourse_row: Tuple[str, Tuple[str, str]], data_dir: str, nlp) -> str:

  document_string, name_time_tuple = discourse_row

  def validate_document_string(document_string: str) -> str:
      """
      Parameters:
      -------------
      document_string: str
        A string document
      
      processed_doc: Document
        A stanza processed document
      """
      removables = re.findall(r'Sr\.as|Sra\.|Srs\.|Sr\.', document_string)
      processed_doc = ' '.join(list(filter(lambda x: x not in removables, re.split(r'\s', document_string)))).strip()
      return nlp(processed_doc)

  validated_doc = validate_document_string(document_string)

  def generate_sentences_from_document(validated_doc: str) -> List[str]:
      return [sent.text for sent in validated_doc.sentences if len(sent.text.split(' ')) > 7]
  
  unique_doc_identifier: str = str(uuid.uuid4())

  unique_doc_id_dict: dict = {'unique_doc_identifier': unique_doc_identifier}

  def parse_sentences_graph(sentence: str, sentence_id: int, doc_identifier: dict, name_time_tuple: Tuple[str, str]) -> dict:
      """
      Parameters:
      --------------
      sentence: str
      sentence_id: int

      Returns:
      --------------
      dict
        A dictionary with the sentence id as key and the data values
      """

      sentence_doc = nlp(sentence)
      lookup = [(word.id, word.text) for sent in sentence_doc.sentences for word in sent.words]
      dependencies = [(word.head if word.head > 0 else word.id, word.id) for sent in sentence_doc.sentences for word in sent.words]
      clean_dependencies = [dependency for dependency in dependencies if dependency[0] != dependency[1]] # Cleaning out the roots
      sentence_root = [word.id for sent in sentence_doc.sentences for word in sent.words if word.head == 0][0]

      sentence_dict = {}

      dep_graph: dict = {'dependency_graph': clean_dependencies}
      lookup_dict: dict = {'lookup': lookup}
      sentence_root_dict: dict = {'root': sentence_root, 'doc_length': len(lookup)}
      sentence_literal: dict = {'sentence_literal': sentence}
      sentence_id: dict = {'sentence_id': sentence_id}
      name_time_dict: dict = {'actor_name': name_time_tuple[0], 'discourse_time': name_time_tuple[1]}

      sentence_data = reduce(lambda dict1, dict2: dict1 | dict2, [lookup_dict, dep_graph, sentence_dict, sentence_root_dict, sentence_literal, doc_identifier, sentence_id, name_time_dict])
      return sentence_data

  document = [parse_sentences_graph(sent, i+1, unique_doc_id_dict, name_time_tuple) for i, sent in enumerate(generate_sentences_from_document(validated_doc))]

  if document:
    with open(data_dir + '/' + f'{unique_doc_identifier}.json', 'w') as json:
      ujson.dump(document, json)
  else: 
    pass


def create_partition_indices(data_length: int):

  range_steps = range(int(np.ceil(data_length/4000)))

  indices = []

  for i in range_steps:
      
      if i == range_steps[-1]:
          first = i * 4000
          last = data_length

      else:
          first = i * 4000
          last = (i+1) * 4000

      indices.append((i, list(range(first, last))))

  return indices


@click.command()
@click.option('--partition_num', type=int)
def export_docs(partition_num):

    partition_num = partition_num - 1

    # Defining paths
    configs = load_config()

    nlp = stanza.Pipeline(lang='pt', processors='tokenize,mwt,pos,lemma,depparse', verbose=False)

    json_data_dir = configs['CONFIG_INFO']['JSON_DATA_DIR']
    data_dir = configs['CONFIG_INFO']['SPEECH_DATA_DIR']

    # Loading the speeches from parquet
    speeches = load_speeches(data_dir)

    filtered_speeches = filter_speeches(speeches)

    discourse_string = create_actor_discourse_string(filtered_speeches)

    # Computing partition number ranges
    total_length = len(discourse_string)

    partitions = create_partition_indices(total_length)

    this_partition = partitions[partition_num][1]

    for i in tqdm(this_partition):
      generate_document_json(discourse_string[i], json_data_dir, nlp)


if __name__ == '__main__':
  export_docs()

