import json
from typing import Dict
from pyspark.sql import DataFrame


def dataset_writer(path: str, dataset: DataFrame, mode: str):
    """
    Writes a PySpark dataframe into a .csv file with header depending on mode chosen
    :param path: the path in which the .csv file will be written
    :param dataset: dataframe containing the data to write inside the .csv file
    :param mode: how to write the .csv file. 'a' for append and 'w' for rewrite
    """
    with open(path, mode) as filehandle:
        if mode == 'w':
            filehandle.write(",".join(dataset.schema.names) + "\n")
        for domain in dataset.collect():
            row = ""
            for i in range(len(dataset.schema.names)):
                row += f"{domain[i]}," if i < len(dataset.schema.names)-1 else f"{domain[i]}\n"
            filehandle.write(row)


def dataset_writer_fasttext(base_path: str, dataset: DataFrame):
    """
    Writes a PySpark dataframe into 3 .txt files. Each file is a dataset that can be used to 
    train a fasttext model. The first file will containe single characters, the second bigramns,
    and the third trigrams. 
    :param base_path: the base path in which the .txt files will be written
    :param dataset: dataframe containing the data to write inside the files
    """
    """with open(f"{base_path}characters.txt", 'w') as filehandle:
        for domain in dataset.collect():
            filehandle.write(f"{domain[-3]}\n")"""
    with open(f"{base_path}bambenekBigrams.txt", 'w') as filehandle:
        for domain in dataset.collect():
            filehandle.write(f"{domain[0]}\n")
    """with open(f"{base_path}trigrams.txt", 'w') as filehandle:
        for domain in dataset.collect():
            filehandle.write(f"{domain[-1]}\n")"""


def metadata_writer(path: str, metadata: Dict):
    """
    Writes metadata associated with a feed into a .json file
    :param path: path in which the file will be written
    :param metadata: dict containing metadata to write
    """
    with open(path, 'w') as m_file:
        json.dump(metadata, m_file)


def metadata_reader(path: str):
    """
    Reads metadata associated with a feed from a Json file
    :param path: json file's path
    :return: a dict containing metadata of the file
    """
    with open(path, 'r') as m_file:
        return json.load(m_file)
