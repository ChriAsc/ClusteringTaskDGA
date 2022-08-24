import os
import datasetWriter
from datetime import date
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *

# se è gia esistente prende lo SparkContext, oppure lo crea
sc = SparkContext.getOrCreate()
# crea una sesione nello spark Context in quanto ci possono essere più session
spark = SparkSession(sc)

# path alla cartella contenente i feed del majestic million
path = f"{os.environ['HOME']}/progettoBDA/MajesticMillion"

# dichiarazione dello schema per la creazione del feed unico
schema = StructType([
    StructField("Domain", StringType(), True),
])

# creazione di un dataframe vuoto con un certo schema che servirà per raccogliere i domini provenienti dai vari
# feed
legit_feed = spark.createDataFrame([], schema)

# raccolta e ordinamento in una lista di tutti i nomi dei file del majestic million
files = os.listdir(path)
files.sort()

# nel caso in cui sia già stato creato un feed e lo si voglia solo arricchire con altri feed di majestic million
# raccolti si controlla se esiste il file dei metadati del feed e usando le informazioni presenti in esso si
# stabiliscono quali sono i file da cui prendere i nomi da aggiungere. Se i metadati non esistono allora i file
# da aggiungere sono tutti i file presenti nella cartella
if os.path.exists(f"{os.environ['HOME']}/progettoBDA/Feed/legitFeedMetadata.json"):
    old_feeds = spark.read.format("csv").option("delimiter", ",").option("header", "true").load(
        f"{os.environ['HOME']}/progettoBDA/Feed/legitFeed.csv")
    metadata = datasetWriter.metadata_reader(f"{os.environ['HOME']}/progettoBDA/Feed/legitFeedMetadata.json")
    start = files.index(metadata['last_feed'])
    files_to_add = files[start+1:]
else:
    metadata = {}
    old_feeds = spark.createDataFrame([], schema)
    files_to_add = files.copy()

# Per ogni file da aggiungere si prendono i nomi dal file e poi eliminando i duplicati si mettono in un
# dataframe che conterrà tutti i domini da aggiungere al feed complessivo
for file in files_to_add:
    df = spark.read.format("csv").option("delimiter", ",").option("header", "true").load(f"{path}/{file}")
    legit_feed = legit_feed.union(df.select('Domain')).dropDuplicates()
# si eliminano dal dataframe dei nomi da aggiungere quelli eventualmente già contenuti nel feed complessivo
legit_feed = legit_feed.subtract(old_feeds)

# istruzioni per la scrittura dei nomi nel feed complessivo e per l'aggiornamento o eventuale
# prima scrittura dei metadati sul file
mode = 'a' if metadata else 'w'
datasetWriter.dataset_writer(f"{os.environ['HOME']}/progettoBDA/Feed/legitFeed.csv", legit_feed, mode)
new_metadata = {"written": date.today().strftime("%Y-%m-%d"),
                "last_feed": files_to_add[-1],
                "num_domains": metadata['num_domains']+legit_feed.count() if metadata else legit_feed.count()}
datasetWriter.metadata_writer(f"{os.environ['HOME']}/progettoBDA/Feed/legitFeedMetadata.json",
                              new_metadata)
