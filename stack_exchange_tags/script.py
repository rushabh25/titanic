import re
import string
import json 

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from BeautifulSoup import BeautifulSoup
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer
from pyspark.ml.classification import NaiveBayes

def removeAlphaNumericCharsAndLower(word):
    word = re.sub(r'\W+', ' ', word)
    return word.lower().strip()


def mapParseLine(line):
    return line.split('","')

def tokenize(text):
    text = text.replace('\n', '')
    tokens = word_tokenize(text)
    lowercased = [t.lower() for t in tokens]
    no_punctuation = []
    stemmed = ''
    PUNCTUATION = set(string.punctuation)
    STOPWORDS = set(stopwords.words('english'))
    STEMMER = PorterStemmer()
    for word in lowercased:
        punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
        no_punctuation.append(punct_removed)
    no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]
    for w in no_stopwords:
        stemmed += w + ' '
    return stemmed.strip()

def removeHTML(text):
    return BeautifulSoup(raw_html).text

replaceNewLine = udf(lambda x: x.replace('\n', ''), StringType())
tokenizedUDF = udf(tokenize, StringType())
removeHTMLUDF = udf(lambda x: (BeautifulSoup(x).text), StringType())

sc._jsc.hadoopConfiguration().set("textinputformat.record.delimiter",'"\n"')
file = sc.textFile('file:///D:/temp/rshah/spark-2.0.1-bin-hadoop2.7/rshah_scripts/exchange_tags/biology.csv')
file = file.map(lambda row: row.split('","'))
fileDF = file.filter(lambda x: (len(x) == 4)).toDF(["id", "title", "content", "tags"])
header_row = fileDF.first()
fileDF = fileDF.rdd.filter(lambda row: row!= header_row).toDF()

fileDF = fileDF.withColumn("tokenizedTitle", tokenizedUDF("title")).withColumn("tokenizedContent", tokenizedUDF(removeHTMLUDF("content")))
fileDF = fileDF.drop("content").drop("title")

tagsDF = fileDF.rdd.map(lambda x: (x[0], x[1].split(' '))).flatMapValues(lambda x: x).toDF(["tags_id", "tags_tags"])
joined_df = tagsDF.join(fileDF, fileDF.id == tagsDF.tags_id, 'inner').drop(tagsDF.tags_id).drop(fileDF.tags)

