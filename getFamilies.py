from pyspark.sql.functions import lit, concat_ws, split
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext


def get_families_distribution(spark: SparkSession, sql: SQLContext, path):
    """
    Detects family distribution of an existent dataset. For each family calculates the number of samples in
    dataset and the percentage on total number of samples
    :param spark: instance of spark session
    :param sql: instance of SQLContext
    :param path: path of feed for studying family distribution
    :return: dataframe containing family distribution
    """
    df = spark.read.format("csv").option("header", "true").load(path)
    df.createOrReplaceTempView("domains")
    family_with_count = sql.sql("SELECT family, COUNT(family) as no_samples FROM domains GROUP BY family")
    tot = df.count()
    family_distribution = family_with_count.withColumn("percentage", (family_with_count['no_samples'] / tot)*100)
    return family_distribution
