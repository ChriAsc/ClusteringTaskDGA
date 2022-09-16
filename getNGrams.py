# imports
from pyspark.sql.functions import lit, split, concat_ws, transform, filter, length
from pyspark.ml.feature import NGram
from pyspark.sql.types import *


def is_bigram(x):
    return BooleanType().toInternal(length(x) > 1)


def is_trigram(x):
    return BooleanType().toInternal(length(x) > 2)


def getNGrams(df):
    df = df.withColumn("array_word", split(df["noDotsDomain"], ''))
    unigrams = NGram(n=1, inputCol="array_word", outputCol="characters")
    bigrams = NGram(n=2, inputCol="array_word", outputCol="bigrams")
    trigrams = NGram(n=3, inputCol="array_word", outputCol="trigrams")
    unigram_domains = unigrams.transform(df)
    bigrams_domains = bigrams.transform(unigram_domains)
    ngrams_domains = trigrams.transform(bigrams_domains)
    ngrams_domains_transformed = ngrams_domains.select("class", "family", "noDotsDomain", "domain",
                                                       concat_ws(" ", "characters").alias("characters"),
                                                       transform("bigrams",
                                                                 lambda x: concat_ws("", split(x, '[ ]'))).alias(
                                                           "bigrams"),
                                                       transform("trigrams",
                                                                 lambda x: concat_ws("", split(x, '[ ]'))).alias(
                                                           "trigrams"))
    processed_domains = ngrams_domains_transformed.select("class", "family", "noDotsDomain", "domain", "characters",
                                                          concat_ws(" ", filter("bigrams", is_bigram)).alias("bigrams"),
                                                          concat_ws(" ", filter("trigrams", is_trigram)).alias(
                                                              "trigrams"))
    return processed_domains

def getNGrams_GARR(df):
    df = df.withColumn("array_word", split(df["noDotsDomain"], ''))
    unigrams = NGram(n=1, inputCol="array_word", outputCol="characters")
    bigrams = NGram(n=2, inputCol="array_word", outputCol="bigrams")
    trigrams = NGram(n=3, inputCol="array_word", outputCol="trigrams")
    unigram_domains = unigrams.transform(df)
    bigrams_domains = bigrams.transform(unigram_domains)
    ngrams_domains = trigrams.transform(bigrams_domains)
    ngrams_domains_transformed = ngrams_domains.select("noDotsDomain", "domain",
                                                       concat_ws(" ", "characters").alias("characters"),
                                                       transform("bigrams",
                                                                 lambda x: concat_ws("", split(x, '[ ]'))).alias(
                                                           "bigrams"),
                                                       transform("trigrams",
                                                                 lambda x: concat_ws("", split(x, '[ ]'))).alias(
                                                           "trigrams"))
    processed_domains = ngrams_domains_transformed.select("noDotsDomain", "domain", "characters",
                                                          concat_ws(" ", filter("bigrams", is_bigram)).alias("bigrams"),
                                                          concat_ws(" ", filter("trigrams", is_trigram)).alias(
                                                              "trigrams"))
    return processed_domains
