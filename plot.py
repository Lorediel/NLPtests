import pandas as pd
import matplotlib.pyplot as plt
import ast
import math
import collections
from transformers import BertTokenizer
import numpy as np

def first_plots():
    # read the tsv dataset
    df = pd.read_csv('./MULTI-Fake-Detective_Task1_Data.tsv', sep='\t')
    df = df.reset_index()
    labels = {
        0: 'Certainly Fake',
        1: 'Probably Fake',
        2: 'Probably Real',
        3: 'Certainly Real',
    }
    total_nums_of_images_for_fakes = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }
    fakes_for_label = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }
    date_labels = {}

    for index, row in df.iterrows():
        date = row["Date"]
        date = str(date)
        if date is not None:
            date = date[:10]
        
        urls = ast.literal_eval(row['Media'])
        num_images = len(urls)
        label = row['Label']
        total_nums_of_images_for_fakes[label] += num_images
        if num_images > 1:
            fakes_for_label[label] += 1


        if date not in date_labels:
            date_labels[date] = {
                0: 0,
                1: 0,
                2: 0,
                3: 0
            }
        date_labels[date][label] += 1

        


    # Total number of images available for each label
    """
    x = list(total_nums_of_images_for_fakes.keys())
    y = list(total_nums_of_images_for_fakes.values())
    """
    # Total number of rows that have more than one image

    x = list(fakes_for_label.keys())
    y = list(fakes_for_label.values())

    """
    # bar plot of dates and label 0
    od = collections.OrderedDict(sorted(date_labels.items()))
    x = list(od.keys())
    y = list(od.values())
    for i in range(4):
        y_l = [item[i] for item in y]
        plt.bar(x, y_l)
        plt.show()
    """
    plt.xticks(x, labels.values())
    # change colors
    plt.bar(x, y, color=['#BB4343', '#FE5E41', '#F1EC88', '#05BF8A'])

    plt.show()

def tweets_articles_lengths():
    bert_tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
    df = pd.read_csv('./MULTI-Fake-Detective_Task1_Data.tsv', sep='\t')
    df = df.reset_index()

    tweet_lengths = []
    article_lengths = []
    tweet_tokens_lengths = []
    article_tokens_lengths = []
    
    for index, row in df.iterrows():
        type = row["Type"]
        text = row["Text"]
        tokens = bert_tokenizer.tokenize(text)
        print(text)
        print(tokens)
        if type == "tweet":
            print(len(text), len(tokens))

            tweet_lengths.append(len(text))
            tweet_tokens_lengths.append(len(tokens))
        else:
            article_lengths.append(len(text))
            article_tokens_lengths.append(len(tokens))
        

    max_lengths = {}
    min_lengths = {}
    avg_lengths = {}

    max_tweet_length = max(tweet_lengths)
    max_article_length = max(article_lengths)
    max_tweet_tokens_length = max(tweet_tokens_lengths)
    max_article_tokens_length = max(article_tokens_lengths)

    max_lengths["tweet"] = max_tweet_length
    max_lengths["article"] = max_article_length
    max_lengths["tweet_tokens"] = max_tweet_tokens_length
    max_lengths["article_tokens"] = max_article_tokens_length

    min_tweet_length = min(tweet_lengths)
    min_article_length = min(article_lengths)
    min_tweet_tokens_length = min(tweet_tokens_lengths)
    min_article_tokens_length = min(article_tokens_lengths)

    min_lengths["tweet"] = min_tweet_length
    min_lengths["article"] = min_article_length
    min_lengths["tweet_tokens"] = min_tweet_tokens_length
    min_lengths["article_tokens"] = min_article_tokens_length

    avg_tweet_length = sum(tweet_lengths) / len(tweet_lengths)
    avg_article_length = sum(article_lengths) / len(article_lengths)
    avg_tweet_tokens_length = sum(tweet_tokens_lengths) / len(tweet_tokens_lengths)
    avg_article_tokens_length = sum(article_tokens_lengths) / len(article_tokens_lengths)

    avg_lengths["tweet"] = avg_tweet_length
    avg_lengths["article"] = avg_article_length
    avg_lengths["tweet_tokens"] = avg_tweet_tokens_length
    avg_lengths["article_tokens"] = avg_article_tokens_length

    return max_lengths, min_lengths, avg_lengths


def tweets_articles_lengths_labels():
    df = pd.read_csv('./MULTI-Fake-Detective_Task1_Data.tsv', sep='\t')
    df = df.reset_index()

    tweet_data = {
        0: [],
        1: [],
        2: [],
        3: []
    }
    article_data = {
        0: [],
        1: [],
        2: [],
        3: []
    }
    for index, row in df.iterrows():
        id = row["ID"]
        type = row["Type"]
        text = row["Text"]
        label = row["Label"]

        text_length = len(text)
        
        if type == "tweet":
            tweet_data[label].append(text_length)

        else:
            article_data[label].append(text_length)
        
    # order each list of the dicts by value
    for label in tweet_data:
        tweet_data[label] = sorted(tweet_data[label])
    for label in article_data:
        article_data[label] = sorted(article_data[label])

    return tweet_data, article_data

tweet_data, article_data = tweets_articles_lengths_labels()

labels = {
        0: 'Certainly Fake',
        1: 'Probably Fake',
        2: 'Probably Real',
        3: 'Certainly Real',
    }

def plot_dict(data):
    # get the length of the longest list
    max_length = max(len(v) for v in data.values())

    # plot the data
    for key, values in data.items():
        # create a list of x-axis values that matches the length of the longest list
        x = list(range(1, max_length+1))
        # pad the values list with None values to match the length of the longest list
        y = values + [None]*(max_length-len(values))
        plt.plot(x, y, label=labels[key])

    # add legend and labels
    plt.legend()
    plt.xlabel("X-axis label")
    plt.ylabel("Y-axis label")
    plt.title("Plot of dictionary values")
    plt.show()


def plot_num_articles_per_label(data):
    labels = {
        0: 'Certainly Fake',
        1: 'Probably Fake',
        2: 'Probably Real',
        3: 'Certainly Real',
    }
    x = list(range(0, 4))
    y = [len(data[label]) for label in data]
    plt.bar(x, y)
    plt.xticks(x, labels.values())
    plt.show()




def plot_dict2(data):
    for key, values in data.items():
        plt.scatter(values, [key]*len(values), label=labels[key])

   
    plt.legend()

    plt.show()

def count_unique(data, min, max):
    unique, counts = np.unique(data, return_counts=True)
    unique_dict = dict(zip(unique, counts))
    for i in range(min, max+1):
        if i not in unique_dict:
            unique_dict[i] = 0
    #take just the values of the dict and put it in a list
    values = list(unique_dict.values())
    return values

def equalObs(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))


def plot_bar_bins(data,title):
    values_label_0 = data[0]
    values_label_1 = data[1]
    values_label_2 = data[2]
    values_label_3 = data[3]

    total_values = np.array(sorted(values_label_0 + values_label_1 + values_label_2 + values_label_3))

    # create 5 bins between the min and max values and put each value in the correct bin
    bins = equalObs(total_values, 4).astype(int)
    print(bins)
    ticks = []
    for b in bins:
        tick = ""
        if b == bins[0]:
            tick = "0 - " + str(bins[0])
        elif b == bins[-1]:
            tick = str(bins[-2]) + " - " + str(bins[-1])
        else:
            tick = str(bins[bins.tolist().index(b)-1]) + " - " + str(bins[bins.tolist().index(b)])
        ticks.append(tick)
    
    values_label_0 = np.digitize(values_label_0, bins)
    values_label_1 = np.digitize(values_label_1, bins)
    values_label_2 = np.digitize(values_label_2, bins)
    values_label_3 = np.digitize(values_label_3, bins)
    

    values_label_0 = count_unique(values_label_0, 1, 5)
    values_label_1 = count_unique(values_label_1, 1, 5)
    values_label_2 = count_unique(values_label_2, 1, 5)
    values_label_3 = count_unique(values_label_3, 1, 5)

    width = 0.2
    # plot the data
    x = np.arange(5)
    
    y0 = values_label_0
    bar0 = plt.bar(x, y0, label=labels[0], width=width)

    y1 = values_label_1
    bar1 = plt.bar(x+width, y1, label=labels[1],width=width)

    y2 = values_label_2
    bar2 = plt.bar(x+width*2, y2, label=labels[2], width=width)

    y3 = values_label_3
    bar3 = plt.bar(x+width*3, y3, label=labels[3], width=width)

    plt.legend( (bar0, bar1, bar2, bar3), (labels[0], labels[1], labels[2], labels[3]))
    plt.xlabel("Lenght of text (characters)")
    plt.ylabel("Number of samples")
    plt.title(title)
    plt.xticks(x+width*1.5, ticks)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('./MULTI-Fake-Detective_Task1_Data.tsv', sep='\t')
    df = df.reset_index()
    min = 100000
    tweet = None
    u = None
    l = None
    tweets_urls = []
    for index, row in df.iterrows():
        id = row["ID"]
        type = row["Type"]
        text = row["Text"]
        label = row["Label"]
        url = row["URL"]
        
        if type == "tweet" and label == 3:
            print(text)
            

   
