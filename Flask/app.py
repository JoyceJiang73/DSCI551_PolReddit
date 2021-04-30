from flask import Flask, jsonify, render_template, request

import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'homepage.html')


#import search
app=Flask(__name__)

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

url ='https://reddit-database-a3721-default-rtdb.firebaseio.com/'

#define functions
def wordcloud_generator(subreddit_input):
    subreddit_filter=requests.get(url+'reddit_post.json?orderBy="subreddit"&equalTo="'+str(subreddit_input)+'"')
    subreddits=json.loads(subreddit_filter.text)
    
    ORG_subreddits=[]
    PERSON_subreddits=[]
    GPE_subreddits=[]
    count=[]
    for x in subreddits:
        try:
            article_id=subreddits[x]['id']
            subreddit_request=requests.get(url+'reddit_ner.json?orderBy="Article"&equalTo="'+str(article_id)+'"')
            subreddit_json=json.loads(subreddit_request.text)
            for i in subreddit_json:
                if subreddit_json[i]['LABEL']=='GPE':
                    GPE_subreddits.append(subreddit_json[i]['TEXT'])
                elif subreddit_json[i]['LABEL']=='PERSON':
                    PERSON_subreddits.append(subreddit_json[i]['TEXT'])
                elif subreddit_json[i]['LABEL']=='ORG':
                    ORG_subreddits.append(subreddit_json[i]['TEXT'])
            count.append(1)
            print(len(count))
        except KeyError:
            continue
            
    #convert list to string and generate
    unique_string=(" ").join([x.replace(' ','') for x in ORG_subreddits])
    wordcloud = WordCloud(width = 400, height = 300, background_color='black',min_word_length=2, ).generate(unique_string)
    plt.figure(figsize=(4,3))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig("Templates/static/images/WordCloud/ORG_subreddits"+".png", bbox_inches='tight')
    plt.savefig("static/images/WordCloud/ORG_subreddits"+".png", bbox_inches='tight')
    plt.savefig("Templates/WordCloud/ORG_subreddits"+".png", bbox_inches='tight')
    plt.savefig("WordCloud/ORG_subreddits"+".png", bbox_inches='tight')
    #plt.show()

    #convert list to string and generate
    unique_string=(" ").join([x.replace(' ','') for x in PERSON_subreddits])
    wordcloud = WordCloud(width = 400, height = 300, background_color='grey',min_word_length=2, ).generate(unique_string)
    plt.figure(figsize=(4,3))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig("Templates/static/images/WordCloud/PERSON_subreddits"+".png", bbox_inches='tight')
    plt.savefig("static/images/WordCloud/PERSON_subreddits"+".png", bbox_inches='tight')
    plt.savefig("Templates/WordCloud/PERSON_subreddits"+".png", bbox_inches='tight')
    plt.savefig("WordCloud/PERSON_subreddits"+".png", bbox_inches='tight')
    #plt.show()
    #plt.close()

    #convert list to string and generate
    unique_string=(" ").join([x.replace(' ','') for x in GPE_subreddits])
    wordcloud = WordCloud(width = 400, height = 300, background_color='white',min_word_length=2, ).generate(unique_string)
    plt.figure(figsize=(4,3))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig("Templates/static/images/WordCloud/GPE_subreddits"+".png", bbox_inches='tight')
    plt.savefig("static/images/WordCloud/GPE_subreddits"+".png", bbox_inches='tight')
    plt.savefig("Templates/WordCloud/GPE_subreddits"+".png", bbox_inches='tight')
    plt.savefig("WordCloud/GPE_subreddits"+".png", bbox_inches='tight')
    #plt.show()
    plt.close()

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
    kmeans = KMeans(k=2,featuresCol="features").setSeed(1)
    kmeans_model=kmeans.fit(data1)
    data1=kmeans_model.transform(data1)
    
    data["prediction"]=data1.select("prediction").toPandas()
    
    return data


@app.route("/")
def index():
    return render_template('homepage.html')

@app.route("/wordcloud", methods=['POST','GET'])
def getvalue():
    if request.method=='POST':
        subreddit_input = request.form['subreddit']
        wordcloud_generator(subreddit_input)
        return render_template('wordcloud_update.html')
    else:
        return render_template('wordcloud.html')

@app.route("/jsondata", methods=['POST','GET'])
def getvalue2():
    if request.method=='POST':
        subreddit_input = request.form['jsondata']
        subreddit_filter=requests.get(url+'reddit_post.json?orderBy="subreddit"&equalTo="'+str(subreddit_input)+'"')
        subreddits_json=json.loads(subreddit_filter.text)
        results=[]
        for x in subreddits_json:
            try:
                results.append(subreddits_json[x])
            except KeyError:
                continue
        jsonfile = json.dumps(subreddits_json, sort_keys = True, indent = 4, separators = (',', ': '))
        return render_template('search_result_update.html', jsonfile=jsonfile)
    else:
        return render_template('search_result.html')


@app.route("/title_topic", methods=['POST','GET'])
def getvalue3():
    if request.method=='POST':
        subreddit_input = request.form['subreddit']
        #subreddit_input = 'World Politics'
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
        data1.show(truncate=False)
        clean_data_udf=udf(clean_data, StringType())
        data1=data1.withColumn("new_title",clean_data_udf("title"))
        data1.show()
        tokenizer = Tokenizer(inputCol="new_title", outputCol="words")
        data1 = tokenizer.transform(data1)
        data1.show()
        remover = StopWordsRemover(inputCol="words", outputCol="rm_words")               
        data1 = remover.transform(data1)
        data1.show()
        hashingTF = HashingTF(inputCol="rm_words", outputCol="rawFeatures", numFeatures=2000)
        data1 = hashingTF.transform(data1)
        data1.show()
        data1.select("rm_words").show(truncate=False)
        data1.select("rawFeatures").show(truncate=False)
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idfModel = idf.fit(data1)
        data1 = idfModel.transform(data1)
        data1.select("features").show(truncate=False)
        kmeans = KMeans(k=2,featuresCol="features").setSeed(1)
        kmeans_model=kmeans.fit(data1)
        data1=kmeans_model.transform(data1)
        data1.select("prediction").show(50)
        data["prediction"]=data1.select("prediction").toPandas()
        print(data["prediction"].value_counts())

        #topic_generator(subreddit_input)
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
        return render_template('title_topic_update.html',topic1_1=topic1_1,topic1_2=topic1_2,topic1_3=topic1_3,topic1_4=topic1_4,topic1_5=topic1_5,
        topic2_1=topic2_1,topic2_2=topic2_2,topic2_3=topic2_3,topic2_4=topic2_4,topic2_5=topic2_5)
    else:
        return render_template('title_topic.html')

@app.route("/sentiment")
def index2():
    return render_template('sentiment.html')

if __name__=="__main__":
    app.run(debug=True)