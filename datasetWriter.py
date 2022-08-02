import json


def dataset_writer(path, dataset, header, mode='a'):
    with open(path, mode) as filehandle:
        filehandle.write(",".join(header) + "\n")
        for domain in dataset.collect():
            row = ""
            for i in range(len(header)):
                row += f"{domain[i]}," if i < len(header)-1 else f"{domain[i]}\n"
            filehandle.write(row)


def metadata_writer(path, metadata):
    with open(path, 'w') as m_file:
        json.dump(metadata, m_file)


def metadata_reader(path):
    with open(path, 'r') as m_file:
        return json.load(m_file)
