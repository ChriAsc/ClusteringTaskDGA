def dataset_writer(path, dataset, header):
    with open(path, 'w') as filehandle:
        filehandle.write(",".join(header) + "\n")
        for domain in dataset.collect():
            row = ""
            for i in range(len(header)):
                row += f"{domain[i]}," if i < len(header)-1 else f"{domain[i]}\n"
            filehandle.write(row)
