import requests
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#import pyspark packages
from pyspark.sql import SparkSession
import pandas as pd
import re
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.clustering import KMeans
#create sparksession
spark = SparkSession.builder.appName('ml').getOrCreate()

#import pyspark
#sc = pyspark.SparkContext(appName="comma22d")
#import packages
#from pyspark.sql import SparkSession, types
#spark=SparkSession.builder.master("loacal").appName('Json File').getOrCreate()


url ='https://reddit-database-a3721-default-rtdb.firebaseio.com/'

#pyspark functinos
# user-defined function for textual preprocessing
def clean_data(text):
    # Convert to lower case
    content=text.lower()
    # Remove all other notations and only keep the words
    content=re.sub('[^a-z ]+', ' ',content)
    # Merge multiple whitespaces to one whitespace
    content=re.sub('\s+', ' ',content)
    return content

def topic_generator(subreddit_input):
    subreddit_filter=requests.get(url+'reddit_post.json?orderBy="subreddit"&equalTo="'+str(subreddit_input)+'"')
    subreddits=json.loads(subreddit_filter.text)
    
    results=[]
    for x in subreddits:
        try:
            results.append(subreddits[x])
        except KeyError:
            continue
    data = pd.DataFrame.from_dict(results, orient='columns')
    data1=spark.createDataFrame(pd.DataFrame(data["title"]))
    
    #text clean
    clean_data_udf=udf(clean_data, StringType())
    data1=data1.withColumn("new_title",clean_data_udf("title"))
    #text tokenizer
    tokenizer = Tokenizer(inputCol="new_title", outputCol="words")
    data1 = tokenizer.transform(data1)
    #stopwords removal
    remover = StopWordsRemover(inputCol="words", outputCol="rm_words")               
    data1 = remover.transform(data1)
    #TFIDF vectorization
    hashingTF = HashingTF(inputCol="rm_words", outputCol="rawFeatures", numFeatures=2000)
    data1 = hashingTF.transform(data1)
    #Document frequency
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(data1)
    data1 = idfModel.transform(data1)
    #Kmeans topic clustering
    kmeans = KMeans(k=4,featuresCol="features").setSeed(1)
    kmeans_model=kmeans.fit(data1)
    data1=kmeans_model.transform(data1)
    
    data["prediction"]=data1.select("prediction").toPandas()
    
    return data

subreddit_input = 'Business'
data=topic_generator(subreddit_input)
topic1=data[data['prediction']==0]['title'].reset_index(drop=True)
topic2=data[data['prediction']==1]['title'].reset_index(drop=True)
topic1_1=topic1[0]
topic1_2=topic1[1]
topic1_3=topic1[2]
topic1_4=topic1[3]
topic1_5=topic1[4]
topic2_1=topic2[0]
topic2_2=topic2[1]
topic2_3=topic2[2]
topic2_4=topic2[3]
topic2_5=topic2[4]
print(topic1_1)