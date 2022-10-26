from datasetWriter import dataset_writer_fasttext
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import *


# se è gia esistente prende lo SparkContext, oppure lo crea
sc = SparkContext.getOrCreate()
# crea una sessione nello spark Context in quanto ci possono essere più session
spark = SparkSession(sc)
# creazione dell'sqlContext per operare query SQL sui dataframe
sqlContext = SQLContext(sc)

bambenek_bigrams = spark.read.format('csv').option("header", "true").load(f"/media/lorenzo/Partizione Dati/progettoBDA/datasets/bambenekBigrams.csv")
bambenek_fasttext = bambenek_bigrams.select("bigrams")
dataset_writer_fasttext("/media/lorenzo/Partizione Dati/progettoBDA/datasets/", bambenek_fasttext)
