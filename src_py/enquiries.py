import re
import toml
import prefect
import datetime
import requests
import xmltodict

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from typing import List, Tuple
from collections import OrderedDict
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from itertools import chain
from functools import reduce
from datetime import datetime
from datetime import timedelta

from prefect import task, Flow, Parameter, unmapped, flatten
from prefect.triggers import all_successful
from prefect.executors import DaskExecutor
from prefect.run_configs import LocalRun

@task 
def load_config():
    # load toml file to dictionary
    configs = toml.load(open('./config/config.toml'))
    return configs

@task(max_retries=3, retry_delay=timedelta(seconds=10))
def retrieve_session_element_ids(url: str):
    """
    Parameters:
    ------------------
    url: str
      The base url for the "Atividade dos Deputados" site

    Returns:
    ------------------
    list
      A list of all the session element ids from the site
    """
    site = requests.get(url)
    site_links = BeautifulSoup(site.content).find_all('div', {'class': 'archive-item'})
    return [link.find_all('a')[0]['id'] for link in site_links][0:4]

@task(max_retries=3, retry_delay=timedelta(seconds=10))
def retrieve_xml_from_session(url: str, session_element_id: str):
    """
    Parameters:
    ------------------
    url: str
      The base url for the Deputies' activities
    session_element_id: str
      A string identifier of the HTML element for the given session

    Returns:
    ------------------
    str: 
      A string with the xml url for the given session
    """

    options = Options()
    options.headless = True

    driver = webdriver.Firefox(options=options)
    driver.set_page_load_timeout(120)
    driver.implicitly_wait(10)
    driver.get(url)

    driver.find_element_by_id('cookieClose').click()

    element = driver.find_element_by_id(session_element_id)
    element.click()
    
    site_links = BeautifulSoup(driver.page_source).find_all('div', {'class': 'archive-item'})
    
    driver.quit()

    xml_links = [link.find_all('a')[0]['href'] for link in site_links if "xml" in link.find_all('a')[0]['href']]

    return xml_links[0]


@task(max_retries=3, retry_delay=timedelta(seconds=10))
def get_xml_response(xml_doc: str):
    """
    Parameters:
    ---------------
    xml_doc: str
      A string with the url of the XML doc
    
    Returns:
    ---------------
    dict_xml: List[OrderedDict]
      A list of OrderedDicts
    """
    xml_response = requests.get(xml_doc)
    xml_content = xml_response.content
    dict_xml = xmltodict.parse(xml_content, process_namespaces=False)
    return dict_xml

@task
def extract_activity_xml(dict_xml: OrderedDict):
  """
  Parameters:
  --------------
  dict_xml: OrderedDict
    The session XML document
  
  Returns:
  --------------
  List[OrderedDict]
    A list of OrderedDicts for each deputy in the session
  """
  deputy_count = len(dict_xml['ArrayOfAtividadeDeputado']['AtividadeDeputado'])
  return [dict_xml['ArrayOfAtividadeDeputado']['AtividadeDeputado'][index] for index in range(deputy_count)]

@task
def extract_deputy_information(xml_dict: OrderedDict):
    """
    Parameters:
    ------------------
    xml_dict: OrderedDict
      An OrderedDict containing the activity of a single deputy

    Returns:
    ------------------
    deputy_data: pd.DataFrame
      A dataframe containing the chained activities of a deputy
    """

    xml_parser_dict = {
        'rel': {
            'relatoresIniciativas': { 
                'pt_gov_ar_wsar_objectos_RelatoresIniciativasOut': ['accDtrel', 'iniTi'] 
                },
            'relatoresPeticoes': {
                'pt_gov_ar_wsar_objectos_RelatoresPeticoesOut': ['pecDtrelf', 'petAspet']
            }
        },
        'ini': {
            'pt_gov_ar_wsar_objectos_IniciativasOut': ['iniId', 'iniTi']
        },
        'req': {
            'pt_gov_ar_wsar_objectos_RequerimentosOut': ['reqDt', 'reqAs']
        },
        'actP': {
            'pt_gov_ar_wsar_objectos_ActividadesParlamentaresOut': ['actDtent', 'actAs']
        }
    }

    logger = prefect.context.get('logger')

    # Defining higher order keys
    possible_keys = ['ini', 'req', 'actP', 'rel']

    deputy_activity = xml_dict['AtividadeDeputadoList']['pt_gov_ar_wsar_objectos_ActividadeOut']
    deputy_name = xml_dict['deputado']['depNomeParlamentar'].title()
    keys_deputy = list(deputy_activity.keys())

    extract_keys = [key for key in keys_deputy if key in possible_keys]

    def extract_key_data(deputy_activity: OrderedDict, key: str):
        """
        Parameters:
        ----------------
        deputy_activity: OrderedDict
            An OrderedDict containing the activity information
        
        key: str
            The key string associated with the type of activity

        Returns:
        ----------------
        deputy_data: pd.DataFrame
            A dataframe with the columns ['activity', 'date', 'name']

        """
        if deputy_activity[key] == None:
            return pd.DataFrame([(np.nan, np.nan), (np.nan, np.nan)], columns=['time', 'text'])
        else:
            if len(deputy_activity[key]) == 1:
                second_order_keys_ls = deputy_activity[key].keys()
            else:
                second_order_keys_ls = list(deputy_activity[key].keys())
            # Get fields associated with a key
            fields = list(chain.from_iterable(xml_parser_dict[key].values()))

            def extract_data(deputy_activity: OrderedDict, key: str, second_order_keys_ls: list):
                activity_list = [deputy_activity[key][second_order_key] for second_order_key in second_order_keys_ls]
                return activity_list
            
            def extract_secondary_keys(chained_activities):
                    present_keys = [key for key in chained_activities[0].keys() if key in fields]
                    return present_keys

            def extract_secondary_key_data(chained_activity: OrderedDict, present_keys: list):
                try:
                    secondary_key_data = [chained_activity[present_key] for present_key in present_keys]
                    if key == 'req':
                        return secondary_key_data[::-1]
                    else:
                        return secondary_key_data
                except KeyError as error:
                    logger.error(error)
                    return None
                
            def reducer(irregular_list: list) -> list:
                reduced_list = [reduce(lambda x, y: {**x,**y}, d.values()) for d in irregular_list]
                objects = []
                for element in reduced_list:
                    if isinstance(element, OrderedDict):
                        objects.append(element)
                    else:
                        objects.extend(element)
                return objects

            def iterate_dict(xml_parser_dict: dict, primary_key: str):
                relator_fields = []
                for key in xml_parser_dict[primary_key].keys():
                    for secondary_key in xml_parser_dict[primary_key][key].keys():
                        relator_fields.append(xml_parser_dict[primary_key][key][secondary_key])
                return relator_fields

            def extract_data_from_keys(data_dictionary: OrderedDict, relator_list: list):
                sub_keys = [all(item in list(data_dictionary.keys()) for item in relator_list[i]) for i in range(len(relator_list))]
                present_keys = list(chain.from_iterable([relator_list[i] for i in range(len(relator_list)) if sub_keys[i] == True]))
                return [data_dictionary[key] for key in present_keys]

            if key != 'rel':
                # Retrieving the list of activities by key
                data = extract_data(deputy_activity, key, second_order_keys_ls)

                if isinstance(data[0], OrderedDict):
                    out = data
                else:
                    out = list(chain.from_iterable(data))

                present_keys = extract_secondary_keys(out)
                secondary_data = [extract_secondary_key_data(item, present_keys) for item in out]
                final_data = list(filter(None, secondary_data))

                if final_data:
                    if not isinstance(final_data[0], list):
                        key_dataframe = pd.DataFrame([final_data], columns=['time', 'text'])
                        key_dataframe['key'] = key
                    else:
                        key_dataframe = pd.DataFrame(final_data, columns=['time', 'text'])
                        key_dataframe['key'] = key
                    return key_dataframe
                else:
                    key_dataframe = pd.DataFrame([(np.nan, np.nan), (np.nan, np.nan)], columns=['time', 'text'])
                    key_dataframe['key'] = key
                    return key_dataframe
            else: # Is run only if key is 'rel'

                # Creates a temporary container with nested OrderedDicts
                temp = extract_data(deputy_activity, key, second_order_keys_ls)

                # Flattens the dictionary structure
                flat_data = reducer(temp)

                # Returns a list of keys for the 'rel' field
                relator_list = iterate_dict(xml_parser_dict, key) 
                
                # Extracts data for matching keys
                extracted_data = [extract_data_from_keys(flat_data[i], relator_list) for i in range(len(flat_data))]

                # Filters out empty cells
                final_data = [data for data in extracted_data if data]

                # Creating dataframe from data with key signature
                if final_data:
                    if not isinstance(final_data[0], list):
                        key_dataframe = pd.DataFrame([final_data], columns=['time', 'text'])
                        key_dataframe['key'] = key
                    else:
                        key_dataframe = pd.DataFrame(final_data, columns=['time', 'text'])
                        key_dataframe['key'] = key
                    return key_dataframe
                else:
                    key_dataframe = pd.DataFrame([(np.nan, np.nan), (np.nan, np.nan)], columns=['time', 'text'])
                    key_dataframe['key'] = key
                    return key_dataframe

    deputy_dataframe = pd.concat([extract_key_data(deputy_activity, key) for key in extract_keys], ignore_index=True)
    deputy_dataframe['name'] = deputy_name

    logger.info(f'Activity successfully parsed.')

    return deputy_dataframe.reset_index(drop=True)


@task(trigger=all_successful)
def reduce_dataframes(dataframes: List[pd.DataFrame]):
    reduced_dataframe = pd.concat(dataframes, ignore_index=True)
    return reduced_dataframe


@task(max_retries=3, retry_delay=timedelta(seconds=10))
def retrieve_initiative_date(initiative_id: str) -> Tuple[str, str]:
    """
    Parameters:
    ------------------
    initiative_id: str
    A string with the initiative identifier

    Returns:
    ------------------
    initiative_date: str
    A string containing the date when an initiative was entered.
    """
    logger = prefect.context.get('logger')

    resource_loc = "https://www.parlamento.pt/ActividadeParlamentar/Paginas/DetalheIniciativa.aspx?BID=" + initiative_id

    res = requests.get(resource_loc, timeout=60)

    initiative_info = BeautifulSoup(res.content)

    if initiative_info.find('span', text='Entrada') == None:
        logger.info(f'No date found for initiative {initiative_id}.')
        return {initiative_id: np.nan}
    else:
        tag = initiative_info.find('span', text='Entrada').parent
        initiative_date = str(datetime.strptime(tag.find('span', text=re.compile(r'\d{4}-\d{1,2}-\d{1,2}')).text, '%Y-%m-%d'))

    logger.info(f'Date retrieved for {initiative_id}.')

    return {initiative_id: initiative_date}


@task
def parse_initiative_ids_from_date_column(speeches_dataframe: pd.DataFrame) -> List[str]:
    time_col = speeches_dataframe['time'].unique().tolist()
    return list(chain.from_iterable([re.findall(r'\d{5,}', str(row)) for row in time_col if re.findall(r'\d{5,}', str(row))]))


@task
def reduce_dictionaries(initiative_mappings: List[dict]):
    return reduce(lambda x, y: x | y, initiative_mappings)


@task
def shape_time_values_in_dataframe(reduced_dataframe: pd.DataFrame, initiative_id_date_mappings: dict) -> pd.DataFrame:
    reduced_dataframe['time'] = pd.to_datetime(reduced_dataframe['time'].replace(initiative_id_date_mappings))
    return reduced_dataframe.dropna(subset=['time']).reset_index(drop=True)

@task
def export_dataframe(dataframe: pd.DataFrame, today: str, data_dir: str):
    export_table = pa.Table.from_pandas(dataframe)
    pq.write_table(export_table, data_dir + f'/deputy_activity_dataframe_{today}.parquet')


with Flow('enquiries-flow') as Flow:
  # Defining paths
  configs = load_config()
  today_string= Parameter('today_string', default=str(datetime.today().date()).replace('-', ''))
  url = Parameter('url', 'https://www.parlamento.pt/Cidadania/Paginas/DAatividadeDeputado.aspx')

  data_dir = configs['CONFIG_INFO']['ACTIVITY_DATA_DIR']

  # Generate HTML ids of sessions
  session_element_ids = retrieve_session_element_ids(url)

  # Get the HTML link for a specific session
  xml_docs = retrieve_xml_from_session.map(unmapped(url), session_element_ids)

  # Retrieve XML response
  xml_dicts = get_xml_response.map(xml_docs)

  # Retrieve actual XML OrderedDicts
  xml_dicts_flat = flatten(extract_activity_xml.map(xml_dicts))

  # Extract deputy information from dict
  deputy_dataframes = extract_deputy_information.map(xml_dicts_flat)

  # Final output
  reduced_dataframe = reduce_dataframes(deputy_dataframes)

  # Retrieving initiative ids from 'time' colum

  initiative_ids = parse_initiative_ids_from_date_column(reduced_dataframe)

  # Retrieving initiative dates from parliament

  initiative_id_date_mappings = retrieve_initiative_date.map(initiative_ids)

  red_id_mappings = reduce_dictionaries(initiative_id_date_mappings)

  # Replacing and parsing values of 'time' column for export
  shaped_dataframe = shape_time_values_in_dataframe(reduced_dataframe, red_id_mappings)

  # Saving to disk
  export_dataframe(shaped_dataframe, today_string, data_dir)

Flow.run_config = LocalRun(working_dir='/home/nfsturm/Dev/Symmachus.jl')
Flow.executor = DaskExecutor()
Flow.register(project_name="sdg-etl")
  
