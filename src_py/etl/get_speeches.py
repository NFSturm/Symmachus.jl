import re
import bs4
import toml
import itertools
import requests
import datetime
import prefect

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from functools import reduce
from toolz.functoolz import pipe
from itertools import chain
from typing import List
from datetime import timedelta
from datetime import datetime
from bs4 import BeautifulSoup
from dateutil.parser import parse

from prefect import task, Flow, Parameter
from prefect.executors import DaskExecutor
from prefect.run_configs import LocalRun
from prefect.triggers import all_finished

@task 
def load_config():
    # load toml file to dictionary
    configs = toml.load(open('./config/config.toml'))
    return configs

@task(max_retries=3, retry_delay=timedelta(seconds=10))
def get_legislatures(base_url: str):
    legislatures = requests.get(base_url)
    soup = BeautifulSoup(legislatures.content, 'html.parser')
    length = len(soup.find_all("td", "clickableCell"))
    return [base_url + '/' + str(number).zfill(2) for number in range(11, length+1) if number > 10]

@task(max_retries=3, retry_delay=timedelta(seconds=10))
def get_sessions_in_legislature(legislature_url):
    legislature = requests.get(legislature_url)
    soup_legislature = BeautifulSoup(legislature.content, 'html.parser')
    table_elements = soup_legislature.find_all("td", {'class', 'clickableCell'})
    
    table_children = []

    for table_element in table_elements:
        table_children.append(table_element.findChildren())
    
    session_num = 0

    session_re = re.compile(r"Sessão(\s*)")

    for child in table_children:
        if re.findall(session_re, str(child)):
            session_num += 1


    return [legislature_url + '/' + str(number).zfill(2) for number in list(range(1, session_num+1))]

@task(max_retries=3, retry_delay=timedelta(seconds=10))
def get_docs_from_session(session_url: str):
    session = requests.get(session_url)
    soup_session = BeautifulSoup(session.content, 'html.parser')
    session_docs = []
    
    for element in soup_session.find_all("td", "clickableCell"):
        session_docs.append(list(element.descendants)) 

    links = [session_docs[i][1].attrs['href'] for i in range(0, len(session_docs)) if type(session_docs[i][1]) != bs4.element.NavigableString]
    
    files = []
    for link in links:
        files.append('/'.join(link.split('/')[-2:]))

    return [session_url + '/' + file_name + '?sft=true' for file_name in files]

@task(max_retries=4, retry_delay=timedelta(seconds=10))
def process_text_files(url: str):
    """
    Parameters:
    -----------------
    url: str
        A url string

    Returns:
    -----------------
    final: pd.DataFrame
        A dataframe with parsed session contents
    """
    ### BLOCK I: Request text data from page

    file = requests.get(url)

    logger = prefect.context.get('logger')
    logger.info(f'Accessing doc {url}')

    file_doc = BeautifulSoup(file.content, 'html.parser')

    interventions_pre = []
    for element in file_doc.find_all('p'):
        interventions_pre.append(element.text)

    def handle_interventions(line: str):
        speak_break_regex = re.compile(r'(?:Presidente):\s*-|(?:Presidente):\s*-|(?:Presidente):\s*—|\):\s*–|\):\s*-|\):\s*—|\):\s*—')
        if re.findall(speak_break_regex, line):
            split_line = re.split(speak_break_regex, line)[-1]
            return split_line.split(' ')
        else:
            return line.split(' ')

    interventions = [intervention for intervention in interventions_pre if len(handle_interventions(intervention)) > 3]

    ### BLOCK II: Finding session executive for splitting the file

    # Import position list

    str_list_pre = [line for line in interventions if line.strip() != '']

    session_exec_ind = []

    session_president_address = re.compile(r'Ex.mo|Ex.ma|Ex.mº|Ex.mª|Ex.ma.|Ex.mo|Ex.ª|Ex.º')

    for counter, element in enumerate(str_list_pre):
        if re.findall(session_president_address, element):
            session_exec_ind.append(counter)
        
    ### BLOCK III: Date extraction

    def make_date(string):
        raw_date = string.split('/')[-1].split('?')[0]
        return str(parse(raw_date))

    session_date = make_date(url)

    ### BLOCK IV: Setting split location for begin of parliamentary session
        
    split_loc = []

    session_exec_re = re.compile(
    r'(A\s*Sr\.ª\s*Presidente\s*?)|(O\s*Sr\.\s*Presidente\s*?)|(O\s*Sr\.\s*Presidente\s*da\s*Assembleia\s*da\s*República\s*?)|(A\s*Sr\.ª\s*Presidente\s*da\s*Assembleia\s*da\s*República\s*?)', re.MULTILINE)

    for counter, element in enumerate(str_list_pre):
        if re.findall(session_exec_re, element):
            split_loc.append(counter)

    ### BLOCK V: Splitting the document

    str_list = str_list_pre[split_loc[0]:]

    ### BLOCK VI: Define removal functions

    def remove_french_quotation_marks(string):
        if re.findall(r'»|»', string):
            return re.sub(r'»|»', '', string)
        else:
            return string

    def remove_continuation_stops(string):
        if '…' in string:
            return re.sub(r'…', '', string)
        else:
            return string

    def remove_meta_info(string):
        expr = re.compile(r'\d+\s*\|\s*[A-Z]{1,2}\s*SÉRIE\s*\-\s*NÚMERO:\s*\d{1,3}\s*\|\s*\d{1,2}\s*DE\s*[A-Z]{4,8}\s*DE\s*\d{4}|[A-Z]+\s*SÉRIE\s*\—\s*NÚMERO\s*\d+', flags=re.IGNORECASE)
        if re.findall(expr, string):
            return re.sub(expr, '', string).strip()
        else: 
            return string

    def remove_morphemes(string):
        return pipe(string, remove_french_quotation_marks, remove_continuation_stops, remove_meta_info)

    def remove_interjections(string):
        pattern = re.compile(r'(Aplausos\s*d[o|e|a]\s*[\w\-\s]+(:\s*—.+[!.?…])?(\s*?[!.?])?)|(Vozes\s*d[o|e|a]\s*[\w\-\s]+(:\s*—.+[!.?…])?(\s*?[!.?])?)|(Risos\s*d[o|e|a]\s*[\w\-\s]+(:\s*—.+[!.?…])?(\s*?[!.?])?)|(Protestos\s*d[o|e|a]\s*[\w\-\s]+(:\s*—.+\.)?(\s*?\.)?)')
        if re.search(pattern, string):
            return re.sub(pattern, '', string)
        else:
            return string

    morphemes = [remove_morphemes(string) for string in str_list]
    clean_morphemes = [remove_interjections(morpheme) for morpheme in morphemes]
    out_ls = [line.strip() for line in clean_morphemes if line.strip() != '']

    def split_sentence(line):
        sentence_pattern = re.compile(r'\s*(?=[O|A]\s*S.+)')
        sentence_splits = re.split(sentence_pattern, line)
        return [split.strip() for split in sentence_splits if split]

    all_lines = list(chain.from_iterable([split_sentence(line) for line in out_ls]))

    def name_start_ind(out_ls):

        name_doc = []

        speak_break_regex = re.compile(r'(?:Presidente):\s*-|(?:Presidente):\s*-|(?:Presidente):\s*—|\)?:\s*–|\)?:\s*-|\)?:\s*—|\)?:\s*—')
        split_regex = re.compile(r'\s*:\s*?—|:\s*?-')

        for count, line in enumerate(out_ls):
            if re.findall(speak_break_regex, line):
                processed = re.split(split_regex, line)[0]
                name_tag = re.findall(r'([O|A]\s*S.+)$', processed)

                if name_tag:
                    name_doc.append([name_tag[0], count])
                else:
                    continue

        return name_doc

    name_doc = name_start_ind(all_lines)

    ### BLOCK IX: Cleaning text
    
    def remove_names_from_line(line):
    
        speak_begin = re.compile(r':(\s*)?—|:(\s*)?-')
    
        if re.findall(speak_begin, line):
            return re.split(speak_begin, line)[-1].strip()
        else:
            return line
        
    clean_txt = list(map(remove_names_from_line, all_lines))
    
    ### BLOCK IX: Creating speaker index
    
    indices = []

    for line_num in range(0, len(name_doc)):
        if line_num == len(name_doc) - 1:
            line_ind = name_doc[line_num][1]
            indices.append([line_ind, line_ind])
        else:
            line_ind = name_doc[line_num][1]
            next_ind = name_doc[line_num + 1][1] - 1
            indices.append([line_ind, next_ind])
            
    def create_index_range(input_list):
        first = input_list[0]
        last = input_list[1]
    
        index_range = range(first, last + 1)
    
        return list(index_range)
    
    ind_ranges = list(map(create_index_range, indices))
    
    ### BLOCK X: Creating name reference for ministers

    def parse_ministers(names):
    
        minister = re.compile(r'(Secretári[a|o]\s*de)|(Primeir[o|a]\s*-)|(Ministr[a|o]\s*d[a|o|e])|(Presidente(\s*)da\s*República)|(Vice\-\s*Primeir[o|a]\s*-)|(Ministrad[o|a]\s*d[a|o|e])')

        ministers = []
        for name in names:
            if re.findall(minister, name):
                ministers.append(name)

        minister_set = set(ministers)
    
        def get_minister_name(line):
            if re.findall(r'(\s*)\(', line):
                return line
            else:
                pass
        
        minister_names = list(filter(None, list(map(get_minister_name, minister_set))))
    
        def create_minister_name_dict(line):
            ls = [item.replace(')', '').strip() for item in line.split('(')]
        
            min_dict3 = {ls[0].replace('  ', ' ').strip(): ls[1].replace('  ', ' ').strip()}
        
            min_dict_out = {}
        
            return min_dict_out | min_dict3
    
        minister_map = list(map(create_minister_name_dict, minister_names))
    
        out_dict = {}
    
        for minister in minister_map:
            out_dict.update(minister)
    
        return out_dict
    
    full_names = [doc[0] for doc in name_doc]

    maps = parse_ministers(full_names)

    ### BLOCK XI: Cleaning names
        
    def clean_name_reference(name_str):
    
        """
        Parameters
        ----------------
        :param name_str: string
            Name string referencing the appellative address of a politician in the data.
    
        Returns
        ----------------
        :returns str
            Subset string without the honorific address
    
        """
        # Removes formal address
        take_away = re.compile(r'O\s*Sr.\s*|A\s*Sr.ª\s*|O\s*S.\s*|A\s*Sr.\s*|A\s*Sr.\s*ª|A\s*Sr.a\s*|A\s*Ser.ª\s*|A\s*St.ª\s*|A\s*S\s*Sr.ª\s*|A\s*Sª')
    
        if re.findall(take_away, name_str):
            pre1 = re.sub(take_away, '', name_str)
        else:
            pre1 = name_str

        # Removes party string in parentheses
        party = re.compile(r'\([A-Z]{2,}(\-[A-Z]{1,})?\)|\([A-Z][a-z]{1,}(\s*)[A-Z][a-z]{3,}\)|\([A-Z|{1,}\)|\([A-Z]{2,}(\/[A-Z]{2,})?\)|\(\s*[a-zA-Z]{1}\s*[a-zA-Z]{1}\s*[a-zA-Z]{1}\)|\(\s*[A-Z]{1}\s*\)')

        if re.findall(party, pre1):
            pre2 = re.sub(party, '', pre1).strip().replace('  ', ' ')
        else:
            pre2 = pre1
        
        not_inscribed = re.compile(r'N\s+[iI]nsc.|Ninsc.')
        
        if re.findall(not_inscribed, pre2):
            out = pre2.split('(')[0].strip()
        else:
            out = pre2
        
        return out

    clean_names_pre = [name for name in list(map(clean_name_reference, full_names))]

    def replace_ministers(line, keys, values):
    
        re_whitespace = re.compile(r'(\s+)')
    
        line_stripped = re_whitespace.sub(' ', line)
    
        return_val = []

        for key_index in range(len(keys)):
            if line_stripped in keys[key_index]:
                return_val.append(values[key_index])
            elif values[key_index] in line_stripped:
                return_val.append(values[key_index])
            else:
                pass

        if return_val:
            return return_val[0]
        else:
            return line

    if maps:
        keys, values = list(zip(*maps.items()))            
        clean_names2 = [replace_ministers(line, keys, values) for line in clean_names_pre]
    else:
        clean_names2 = clean_names_pre
        
    # Since session executives and secretaries are not part of the discourse, their names are replaced with a sentinel
    def set_admin_staff_sentinel(line):
        if re.findall(r'(O\s*Sr.\s*)?Presidente(\s*)?(\(|$)?|(O\s*Sr.\s*)?Secretário(\s*)?(\(|$)?|(A\s*Sr.ª\s*)?Secretária(\s*)?(\(|$)?', line):
            return '&%&'
        else:
            return line

    clean_names_w_sentinel = [set_admin_staff_sentinel(name) for name in clean_names2]

    ### BLOCK XII: Formatting names

    # Some names follow the shape of "firstnameLastname", i.e. without space. 

    def split_name_morphemes(line):
        if line != '&%&':
            splits = re.split(r'(?=[A-Z])', line)
            return [split.strip() for split in splits if split != '']
        else:
            return line

    names_split = [split_name_morphemes(name) for name in clean_names_w_sentinel]

    def recombine_name_morphemes(split_name):
        if split_name == '&%&':
            return split_name
        else: 
            return reduce(lambda w1, w2: w1 + ' ' + w2, split_name)
        
    names_recombined = [recombine_name_morphemes(clean_name) for clean_name in names_split]

    def remove_literal_residuals(line):
        if re.findall(r'–|—|\?|\.|\!', line):
            return re.sub(r'–|—|\?|\.|\!', '', line)
        else:
            return line

    names_ls = [remove_literal_residuals(name) for name in names_recombined]
        
    names_indexed = list(zip(names_ls, ind_ranges))
    
    ### BLOCK XIII: Creating DataFrame for output
    
    def join_index_and_name(name_index_tuple, clean_txt):
        """
        Params:
        ------------
        name_index_tuple: tuple
          A tuple containing the name of the speaker and the speaking indices
      
        clean_txt: list, list      
          A list containing the text of a speaker
      
        Returns:
        ------------
        out_list: list
          A list of tuples containing the speaker name and the combined speech
    
        """
    
        indices = name_index_tuple[1]
        speaker_name = name_index_tuple[0]
        out_tup = (speaker_name, ' '.join([clean_txt[i] for i in indices]))
    
        return out_tup
    
    frame_base = [join_index_and_name(tup, clean_txt) for tup in names_indexed]
    
    pre_final = pd.DataFrame(frame_base, columns = ['name', 'text'])
    pre_final.loc[:, 'time'] = session_date

    pre_final2 = pre_final.loc[pre_final['name'] != '&%&', :].reset_index(drop=True)    

    final = pre_final2.loc[:, ['text', 'time', 'name']]

    final['data_source'] = 'assembleia'
    
    return final.reset_index(drop=True)

@task
def list_reduce(results: List[list]):
    return list(itertools.chain.from_iterable(results))

@task(trigger=all_finished)
def reduce_dataframes(dataframes: List[pd.DataFrame]):
    return pd.concat(dataframes).reset_index(drop=True)

@task
def export_dataframe(dataframe: pd.DataFrame, today: str, data_dir: str):
    export_table = pa.Table.from_pandas(dataframe)
    pq.write_table(export_table, data_dir + f'/ddr_dataframe_{today}.parquet')

@task
def generate_logs(num_legislatures: int, sessions: List[list], docs: List[list], state_path: str):
    sessions_in_legislature = [len(legislature) for legislature in sessions]

    indices = [index-1 for index in np.cumsum([len(legislature) for legislature in sessions])]
    ranges_ls = [(indices[i]+1, indices[i+1]) for i in range(len(indices)) if i != len(indices)-1]
    ranges_ls.insert(0, (0, ranges_ls[0][0]-1))

    doc_len = [len(doc) for doc in docs]
    doc_length_in_session = [doc_len[indices[0]:indices[1]+1] for indices in ranges_ls]

    log_data = pd.DataFrame(list(zip(list(range(7, len(num_legislatures)+8)), sessions_in_legislature, doc_length_in_session)), columns=['legislature', 'session_num', 'entries'])

    log_data['state_date'] = pd.Timestamp(datetime.now().strftime("%d-%b-%Y %H:%M:%S"))

    log_data.to_csv(state_path + f'/logs_ddr_state.csv', index=False)


with Flow('ddr-ingestion') as Flow:

    base_url = Parameter('url', default='https://debates.parlamento.pt/catalogo/r3/dar/01')
    today_string= Parameter('today_string', default=str(datetime.today().date()).replace('-', ''))

    # Defining paths
    configs = load_config()

    state_path = configs['CONFIG_INFO']['STATE_DIR']
    data_dir = configs['CONFIG_INFO']['ACTIVITY_DATA_DIR']

    # Retrieving length of legislatures
    number_of_legislatures = get_legislatures(base_url)

    # Generate complex object for the scraper (Source of problems)
    sessions = get_sessions_in_legislature.map(number_of_legislatures)

    session_list = list_reduce(sessions)

    # Extract session information 
    docs = get_docs_from_session.map(session_list)

    # Storing the state. The task is set here to be able to readily inspect the data in case  of problems.
    generate_logs(number_of_legislatures, sessions, docs, state_path)

    # Retrieving list of session urls for processing
    document_links_list = list_reduce(docs)

    # Processing PDF Byte stream into pd.DataFrame
    processed_dataframes = process_text_files.map(document_links_list)

    # Final output
    reduced_dataframe = reduce_dataframes(processed_dataframes)

    # Saving to disk
    export_dataframe(reduced_dataframe, today_string, data_dir)

Flow.run_config = LocalRun(working_dir='/home/nfsturm/Dev/Symmachus.jl')
Flow.executor = DaskExecutor()
Flow.register(project_name="sdg-etl")