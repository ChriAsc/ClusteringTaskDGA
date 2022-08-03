import json
from typing import Dict
from pyspark.sql import DataFrame


def dataset_writer(path: str, dataset: DataFrame, mode: str):
    """
    Writes a PySpark dataframe into a .csv file with header depending on mode chosen
    :param path: the path in which write the .csv file
    :param dataset: dataframe containing data to write inside .csv file
    :param mode: how to write the .csv file. a for append and w for rewrite
    """
    with open(path, mode) as filehandle:
        if mode == 'w':
            filehandle.write(",".join(dataset.schema.names) + "\n")
        for domain in dataset.collect():
            row = ""
            for i in range(len(dataset.schema.names)):
                row += f"{domain[i]}," if i < len(dataset.schema.names)-1 else f"{domain[i]}\n"
            filehandle.write(row)


def metadata_writer(path: str, metadata: Dict):
    """
    Writes metadata associated with a feed into a .json file
    :param path: path in which write the file
    :param metadata: dict containing metadata to write
    """
    with open(path, 'w') as m_file:
        json.dump(metadata, m_file)


def metadata_reader(path: str):
    """
    Reads metadata associated with a feed from Json file
    :param path: json file's path
    :return: a dict containing metadata of file
    """
    with open(path, 'r') as m_file:
        return json.load(m_file)
