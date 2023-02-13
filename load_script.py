import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from datetime import datetime
import os
import sys

import snscrape.modules.twitter as sntwitter
from newspaper import Article

MEDIA_DIR = "./Media/"

def download_images_tweets(tweet):
    urls = [media.fullUrl for media in tweet.media]
    media_list = []
    filename = str(tweet.id)+"_"
    if len(urls) > 1:
        for i, url in enumerate(urls, 1):
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            savename = f"{filename+str(i)}.jpg"
            img.save(MEDIA_DIR+savename)
            media_list.append(savename)
    elif len(urls) == 1:
        response = requests.get(urls[0])
        img = Image.open(BytesIO(response.content)).convert("RGB")
        savename = f"{filename}.jpg"
        img.save(MEDIA_DIR+savename)
        media_list.append(savename)
    else:
        media_list = []
    return media_list

        
def get_tweet(twid):
    tweet = [tweet for tweet in sntwitter.TwitterTweetScraper(twid).get_items()][0]
    try:
        media_list = download_images_tweets(tweet)
    except Exception:
        raise
    return [tweet.id, tweet.url, tweet.date, "tweet", tweet.rawContent, media_list] 
            

def download_top_image_article(article, artId):
    response = requests.get(article.top_image)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    savename = f"{artId}_.jpg"
    img.save(MEDIA_DIR+savename)
    return [savename]
    
def get_article(url, artId):
    
    article = Article(url)
    article.download()
    article.html 
    article.parse()
    text = article.text
    date = article.publish_date
    media_list = download_top_image_article(article, artId)
    return [artId, url, date, "article", text, media_list]

def get_data(row):
    if row.Type == "tweet":
        try:
            data = get_tweet(row.ID)
        except:
            return None
    elif row.Type == "article":
        try:
            data = get_article(row.URL, row.ID)
        except:
            return None
    return data+[row.Label]

def main(task_id_file):
    
    if not os.path.exists(MEDIA_DIR):
        os.makedirs(MEDIA_DIR)

    id_data = pd.read_csv(task_id_file, sep= "\t")

    data = [get_data(row) for row in tqdm(id_data.itertuples(index = False), total = id_data.shape[0])]
    data = [e for e in data if e is not None]

    columns = ["ID","URL","Date","Type","Text","Media","Label"]
    dataframe = pd.DataFrame(data, columns = columns)

    outfile = task_id_file.replace("IDs.tsv", "Data.tsv")
    dataframe.to_csv(outfile, sep ="\t", index = False)

if __name__ == "__main__":
    main(sys.argv[1])