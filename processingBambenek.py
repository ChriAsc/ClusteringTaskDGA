import os
import datasetWriter
from datetime import date
from getFamilies import get_families_distribution
from pyspark.context import SparkContext
from pyspark.sql.functions import lit, concat_ws, split, when, col
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *


# se è gia esistente prende lo SparkContext, oppure lo crea
sc = SparkContext.getOrCreate()
# crea una sessione nello spark Context in quanto ci possono essere più session
spark = SparkSession(sc)
# creazione dell'sqlContext per operare query SQL sui dataframe
sqlContext = SQLContext(sc)

# schema per la lettura dei dati dal feed bambenek
schema = StructType([
    StructField("domain", StringType(), True),
    StructField("description", StringType(), True),
    StructField("date", StringType(), True),
    StructField("link", StringType(), True),
])

# schema per la scrittura dei dati sul feed complessivo dei domini malevoli
write_schema = StructType([
    StructField("domain", StringType(), True),
    StructField("description", StringType(), True)
])

# raccolta e ordinamento in una lista di tutti i nomi dei file del bambenek feed
bambenek_feeds = os.listdir(f"{os.environ['HOME']}/progettoBDA/DGAFeed/")
bambenek_feeds.sort()

# nel caso in cui sia già stato creato un feed e lo si voglia solo arricchire con altri feed di bambenek
# raccolti si controlla se esiste il file dei metadati del feed e usando le informazioni presenti in esso si
# stabiliscono quali sono i file da cui prendere i nomi da aggiungere. Se i metadati non esistono allora i file
# da aggiungere sono tutti i file presenti nella cartella
if os.path.exists(f"{os.environ['HOME']}/progettoBDA/Feed/bambenekFeedMetadata.json"):
    old_feeds = spark.read.format("csv").option("delimiter", ",").option("header", "true").load(
        f"{os.environ['HOME']}/progettoBDA/Feed/bambenekFeed.csv")
    metadata = datasetWriter.metadata_reader(f"{os.environ['HOME']}/progettoBDA/Feed/bambenekFeedMetadata.json")
    start = bambenek_feeds.index(metadata['last_feed'])
    feeds_to_add = bambenek_feeds[start+1:]
else:
    metadata = {}
    old_feeds = spark.createDataFrame([], write_schema)
    feeds_to_add = bambenek_feeds.copy()

# creazione di un dataframe vuoto per ospitare i dati provenienti dai nuovi feed di bambenek
evil_feed = spark.createDataFrame([],schema)

# Per ogni file da aggiungere si prendono i nomi e tutte le altre informazioni dal file eliminando le righe
# di intestazione. Dopodichè si estrae dalle informazioni associato al nome la famiglia a cui esso corrisponde
# mediante informazioni note e corrispondenze tra vari alias. Infine si ottiene  un dataframe che conterrà
# tutti i domini da aggiungere al feed complessivo di cui riportiamo nome e famiglia
for feed in feeds_to_add:
    df = spark.read.format('csv').load(f"{os.environ['HOME']}/progettoBDA/DGAFeed/{feed}", schema=schema)
    evil_feed = evil_feed.union(df.filter(df['description'].isNotNull()))
evil_feed = evil_feed.withColumn("family", split(evil_feed['link'], '[/.]')[6])
evil_feed = evil_feed.withColumn("family", when(col('family') == 'cl', 'cryptolocker')
                                 .when(col('family') == 'wiki25', 'cryptolocker')
                                 .when(col('family') == 'bebloh', 'shiotob')
                                 .when(col('family') == 'ptgoz', 'zeus-newgoz').otherwise(col('family')))
evil_feed = evil_feed.select("domain", "family")

# si eliminano eventuali duplicati che esistono tra i file processati
evil_feed = evil_feed.dropDuplicates()

# si eliminano dal dataframe dei nomi da aggiungere quelli già presenti nel feed complessivo
evil_feed = evil_feed.subtract(old_feeds)

# istruzioni per la scrittura dei nomi nel feed complessivo e per l'aggiornamento o eventuale
# prima scrittura dei metadati sul file
mode = 'a' if metadata else 'w'
datasetWriter.dataset_writer(f"{os.environ['HOME']}/progettoBDA/Feed/bambenekFeed.csv", evil_feed, mode)
new_metadata = {"written": date.today().strftime("%Y-%m-%d"),
                "last_feed": feeds_to_add[-1],
                "num_domains": metadata['num_domains']+evil_feed.count() if metadata else evil_feed.count()}
datasetWriter.metadata_writer(f"{os.environ['HOME']}/progettoBDA/Feed/bambenekFeedMetadata.json",
                              new_metadata)
# determinazione del numero di campioni per ogni famiglia rispetto al totale dei campioni
family_distribution = get_families_distribution(spark, sqlContext, f"{os.environ['HOME']}/progettoBDA/Feed/bambenekFeed.csv")
family_distribution.show(100)
"""
ISTRUZIONI PER STUDIARE LE SOVRAPPOSIZIONI TRA I DOMINI DI ALCUNE FAMIGLIE
old_feeds = spark.read.format("csv").option("delimiter", ",").option("header", "true").load(f"{os.environ['HOME']}//progettoBDA/Feed/bambenekFeed.csv")
old_feeds.createOrReplaceTempView('evil_feed')
studio = sqlContext.sql("SELECT domain, concat(first(family), ' ' ,last(family)) as sovrapposizioni, COUNT(domain) FROM evil_feed GROUP BY domain HAVING COUNT(domain) >= 2")
studio.createOrReplaceTempView('studio')
sovrapposizioni = sqlContext.sql("SELECT sovrapposizioni, COUNT(sovrapposizioni) FROM studio GROUP BY sovrapposizioni")"""