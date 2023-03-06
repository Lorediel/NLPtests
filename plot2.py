
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from transformers import AutoModel, AutoTokenizer



label_names = {
        0: 'Certainly Fake',
        1: 'Probably Fake',
        2: 'Probably Real',
        3: 'Certainly Real',
    }

colors = {
    'yellow': '#D4B483',
    'red': '#C1666B',
    'green': '#48A9A6',
    'blue': '#4281A4'
}

"""
------------------ PLOT 1 ------------------
Distribution of labels
"""
def label_distribution():
    df = pd.read_csv('./MULTI-Fake-Detective_Task1_Data.tsv', sep='\t').drop_duplicates(keep="first", ignore_index=True)
    df = df.reset_index()
    samples_for_label = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }
    for index, row in df.iterrows():
        label = row['Label']
        samples_for_label[label] += 1

    # use computer modern font
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'STIXGeneral'

    # increase font size
    plt.rcParams.update({'font.size': 18})
    
    plt.title("Label distribution" )

    ax = plt.gca()
    ymax = max(samples_for_label.values())
    ax.set_ylim([0, ymax + 40])
    barplot = plt.bar(label_names.values(), samples_for_label.values(), color=[colors['red'], colors['yellow'], colors['green'], colors['blue']])
    # add values to the bars
    plt.bar_label(barplot)

    plt.show()
    return samples_for_label

"""
------------------ PLOT 2 ------------------
Distribution of labels for tweets and articles
"""

def type_label_distribution(type):
    if type != 'tweet' and type != 'article':
        raise ValueError('type must be "tweet" or "article"')
    
    # use computer modern font
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'STIXGeneral'

    # increase font size
    plt.rcParams.update({'font.size': 18})

    df = pd.read_csv('./MULTI-Fake-Detective_Task1_Data.tsv', sep='\t').drop_duplicates(keep="first", ignore_index=True)
    df = df.reset_index()
    samples_for_label = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }
    

    for index, row in df.iterrows():
        label = row['Label']
        t = row["Type"]
        if type == t:
            samples_for_label[label] += 1

    ax = plt.gca()
    ymax = max(samples_for_label.values())
    ax.set_ylim([0, ymax + 40])

    plt.title(type.capitalize() + "s label distribution" )
    barplot = plt.bar(label_names.values(), samples_for_label.values(), color=[colors['red'], colors['yellow'], colors['green'], colors['blue']])
    plt.bar_label(barplot)
    plt.show()
    return samples_for_label

"""
------------------ PLOT 3 ------------------
Length of the text
"""


def tweets_articles_lengths_for_labels():
    df = pd.read_csv('./MULTI-Fake-Detective_Task1_Data.tsv', sep='\t').drop_duplicates(keep="first", ignore_index=True)
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

        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
        text_length = len(tokenizer.tokenize(text))
        #text_length = len(text)
        
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

def count_unique(data, min, max):
    unique, counts = np.unique(data, return_counts=True)
    
    unique_dict = dict(zip(unique, counts))
    
    for i in range(min, max+1):
        if i not in unique_dict:
            unique_dict[i] = 0
    #order the dict by key
    unique_dict = dict(sorted(unique_dict.items()))
    #take just the values of the dict and put it in a list
    values = list(unique_dict.values())
    return values

def equalObs(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))

def include_right_edge(values, num_bins):
    for i in range(len(values)):
        if values[i] == num_bins+1:
            values[i] = values[i] - 1
    return values


def length_bins(num_bins, type):
    # use computer modern font
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'STIXGeneral'

    # increase font size
    plt.rcParams.update({'font.size': 18})

    tweet_data, article_data = tweets_articles_lengths_for_labels()
    if type == "tweet":
        data = tweet_data
    elif type == "article":
        data = article_data
    else:
        raise ValueError("type must be 'tweet' or 'article'")

    values_label_0 = data[0]
    values_label_1 = data[1]
    values_label_2 = data[2]
    values_label_3 = data[3]

    total_values = np.array(sorted(values_label_0 + values_label_1 + values_label_2 + values_label_3))

    # divide total values in lists of equal length
    

    bins = equalObs(total_values, num_bins).astype(int)
    ticks = []
    
    for i in range(len(bins)):
        if i != len(bins)-1:
            tick = str(bins[i]) + "-" + str(bins[i+1])
        
            ticks.append(tick)
    bins = np.array(bins)
    
    
    values_label_0 = include_right_edge(np.digitize(values_label_0, bins), num_bins)
    
    values_label_1 = include_right_edge(np.digitize(values_label_1, bins), num_bins)
    
    values_label_2 = include_right_edge(np.digitize(values_label_2, bins), num_bins)
    
    values_label_3 = include_right_edge(np.digitize(values_label_3, bins), num_bins)
    
    print(values_label_0)
    values_label_0 = count_unique(values_label_0, 1, num_bins)

    values_label_1 = count_unique(values_label_1, 1, num_bins)
    values_label_2 = count_unique(values_label_2, 1, num_bins)
    values_label_3 = count_unique(values_label_3, 1, num_bins)

    width = 0.2
    # plot the data
    x = np.arange(1, num_bins+1)
    
    
    y0 = values_label_0
    bar0 = plt.bar(x, y0, label=label_names[0], width=width, color=colors['red'])

    y1 = values_label_1
    bar1 = plt.bar(x+width, y1, label=label_names[1],width=width, color=colors['yellow'])

    y2 = values_label_2
    bar2 = plt.bar(x+width*2, y2, label=label_names[2], width=width, color=colors['green'])

    y3 = values_label_3
    bar3 = plt.bar(x+width*3, y3, label=label_names[3], width=width, color=colors['blue'])

    plt.bar_label(bar0)
    plt.bar_label(bar1)
    plt.bar_label(bar2)
    plt.bar_label(bar3)
    #change legend font size


    legend = plt.legend( (bar0, bar1, bar2, bar3), (label_names[0], label_names[1], label_names[2], label_names[3]), loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=4, prop={'size': 14})
    legend.get_frame().set_alpha(None)
    # put the legend out of the figure
    plt.xlabel("Lenght of text (characters)")
    plt.ylabel("Number of samples")
    if type == "tweet":
        title = "Tweets length distribution"
    else:
        title = "Articles length distribution"
    plt.title(title, loc='center', pad=30)
    plt.xticks(x+width*1.5, ticks)
    plt.show()

"""
---------------------- PLOT 4  ----------------------
Number of images per label
"""
def avg_images_per_label(type = None):
    if type != None and type != "tweet" and type != "article":
        raise ValueError("type must be 'tweet' or 'article'")
    df = pd.read_csv('./MULTI-Fake-Detective_Task1_Data.tsv', sep='\t').drop_duplicates(keep="first", ignore_index=True)
    df = df.reset_index()

    samples_label_0 = 0
    samples_label_1 = 0
    samples_label_2 = 0
    samples_label_3 = 0

    num_images_label_0 = 0
    num_images_label_1 = 0
    num_images_label_2 = 0
    num_images_label_3 = 0

    for index, row in df.iterrows():
        label = row['Label']
        t = row["Type"]
        urls = ast.literal_eval(row['Media'])
        if type != None:
            if t != type:
                continue
        if label == 0:
            samples_label_0 += 1
            num_images_label_0 += len(urls)
        elif label == 1:
            samples_label_1 += 1
            num_images_label_1 += len(urls)
        elif label == 2:
            samples_label_2 += 1
            num_images_label_2 += len(urls)
        elif label == 3:
            samples_label_3 += 1
            num_images_label_3 += len(urls)

    #print(str(round(num_images_label_0/samples_label_0, 3)) + "\t" + str(round(num_images_label_1/samples_label_1, 3)) + "\t" + str(round(num_images_label_2/samples_label_2, 3)) + "\t" + str(round(num_images_label_3/samples_label_3, 3)))
    
    print("Label 0: ", num_images_label_0/samples_label_0)
    print("Label 1: ", num_images_label_1/samples_label_1)
    print("Label 2: ", num_images_label_2/samples_label_2)
    print("Label 3: ", num_images_label_3/samples_label_3)

# average percentage of capital letters per label
def count_letters_df(type = None):
    if type != None and type != "tweet" and type != "article":
        raise ValueError("type must be 'tweet' or 'article'")
    df = pd.read_csv('./MULTI-Fake-Detective_Task1_Data.tsv', sep='\t').drop_duplicates(keep="first", ignore_index=True)
    df = df.reset_index()

    # use computer modern font
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'STIXGeneral'

    # increase font size
    plt.rcParams.update({'font.size': 18})

    samples_label_0 = 0
    samples_label_1 = 0
    samples_label_2 = 0
    samples_label_3 = 0

    percent_capital_letters_label_0 = 0
    percent_capital_letters_label_1 = 0
    percent_capital_letters_label_2 = 0
    percent_capital_letters_label_3 = 0

    for index, row in df.iterrows():
        text = row['Text']
        num_letters = len(text)
        num_capital_letters = count_capital_letters(text)
        label = row['Label']
        t = row["Type"]
        percent_capital_letters = num_capital_letters/num_letters
        if type != None:
            if t != type:
                continue
        if label == 0:
            samples_label_0 += 1
            percent_capital_letters_label_0 += percent_capital_letters
        elif label == 1:
            samples_label_1 += 1
            percent_capital_letters_label_1 += percent_capital_letters
        elif label == 2:
            samples_label_2 += 1
            percent_capital_letters_label_2 += percent_capital_letters
        elif label == 3:
            samples_label_3 += 1
            percent_capital_letters_label_3 += percent_capital_letters

    vals_for_label = {
        0: round(percent_capital_letters_label_0/samples_label_0, 3),
        1: round(percent_capital_letters_label_1/samples_label_1, 3),
        2: round(percent_capital_letters_label_2/samples_label_2, 3),
        3: round(percent_capital_letters_label_3/samples_label_3, 3)
    }
    
    ax = plt.gca()
    ymax = max(vals_for_label.values())
    ax.set_ylim([0, ymax + 0.05])
    
    if type == None:
        plt.title("Average percentage of capital letters per label")
    else:
        plt.title("Average percentage of capital letters per label (" + type + ")")
    barplot = plt.bar(label_names.values(), vals_for_label.values(), color=[colors['red'], colors['yellow'], colors['green'], colors['blue']])
    plt.bar_label(barplot)

    plt.show()

# percentage of capital letters per label considering total text

def count_letters_df2(type = None):
    if type != None and type != "tweet" and type != "article":
        raise ValueError("type must be 'tweet' or 'article'")
    df = pd.read_csv('./MULTI-Fake-Detective_Task1_Data.tsv', sep='\t').drop_duplicates(keep="first", ignore_index=True)
    df = df.reset_index()

    total_letters_label_0 = 0
    total_letters_label_1 = 0
    total_letters_label_2 = 0
    total_letters_label_3 = 0

    total_capital_letters_label_0 = 0
    total_capital_letters_label_1 = 0
    total_capital_letters_label_2 = 0
    total_capital_letters_label_3 = 0


    for index, row in df.iterrows():
        text = row['Text']
        num_letters = len(text)
        num_capital_letters = count_capital_letters(text)
        label = row['Label']
        t = row["Type"]
        if type != None:
            if t != type:
                continue
        if label == 0:
            total_letters_label_0 += num_letters
            total_capital_letters_label_0 += num_capital_letters
        elif label == 1:
            total_letters_label_1 += num_letters
            total_capital_letters_label_1 += num_capital_letters
            
        elif label == 2:
            total_letters_label_2 += num_letters
            total_capital_letters_label_2 += num_capital_letters
            
        elif label == 3:
            total_letters_label_3 += num_letters
            total_capital_letters_label_3 += num_capital_letters

    vals_for_label = {
        0: round(total_capital_letters_label_0/total_letters_label_0,3),
        1: round(total_capital_letters_label_1/total_letters_label_1,3),
        2: round(total_capital_letters_label_2/total_letters_label_2,3),
        3: round(total_capital_letters_label_3/total_letters_label_3,3)
    }

    return vals_for_label

            
def count_capital_letters(text):
    count = 0
    for c in text:
        if c.isupper():
            count += 1
    return count

def plot_scatter_length_capital(t = None):
    if t != None and t != "tweet" and t != "article":
        raise ValueError("t must be 'tweet' or 'article'")
    df = pd.read_csv('./MULTI-Fake-Detective_Task1_Data.tsv', sep='\t').drop_duplicates(keep="first", ignore_index=True)
    df = df.reset_index()

    len_texts_label_0 = []
    len_texts_label_1 = []
    len_texts_label_2 = []
    len_texts_label_3 = []

    percent_capital_letters_label_0 = []
    percent_capital_letters_label_1 = []
    percent_capital_letters_label_2 = []
    percent_capital_letters_label_3 = []

    
    
    for index, rows in df.iterrows():
        if t != None:
            if rows["Type"] != t:
                continue
        label = rows["Label"]
        text = rows["Text"]
        if label == 0:
            len_texts_label_0.append(len(text))
            num_capital_letters = count_capital_letters(text)
            percent_capital_letters_label_0.append(num_capital_letters/len(text))
        elif label == 1:
            len_texts_label_1.append(len(text))
            num_capital_letters = count_capital_letters(text)
            percent_capital_letters_label_1.append(num_capital_letters/len(text))
        elif label == 2:
            len_texts_label_2.append(len(text))
            num_capital_letters = count_capital_letters(text)
            percent_capital_letters_label_2.append(num_capital_letters/len(text))
        elif label == 3:
            len_texts_label_3.append(len(text))
            num_capital_letters = count_capital_letters(text)
            percent_capital_letters_label_3.append(num_capital_letters/len(text))

    len_texts = [len_texts_label_0, len_texts_label_1, len_texts_label_2, len_texts_label_3]
    percent_capital_letters = [percent_capital_letters_label_0, percent_capital_letters_label_1, percent_capital_letters_label_2, percent_capital_letters_label_3]
    colors = ["#C1666B", "#D4B483", "#48A9A6", "#4281A4"]
    plt.title("Text length against Percentage of capital letters per label" + (" (" + t + ")" if t != None else ""))
    for i in range(4):
        plt.scatter(len_texts[i], percent_capital_letters[i], label=label_names[i], color=colors[i])
    plt.xlabel("Length of text")
    plt.ylabel("Percent of capital letters")
    plt.legend()
    plt.show()  
    
if __name__ == "__main__":
    df = pd.read_csv('./MULTI-Fake-Detective_Task1_Data.tsv', sep='\t').drop_duplicates(keep="first", ignore_index=True)
    df = df.reset_index()
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
    max_len = 0
    for i, rows in df.iterrows():
        if rows["Type"] == "article":
            tokenized = tokenizer.tokenize(rows["Text"])
            if len(tokenized) > max_len:
                max_len = len(tokenized)
    print(max_len)
    """
    texts  = []
    texts_urls = []
    duplicated_urls = []
    duplicates = []
    labels = []
    duplicated_labels = []
    for i, rows in df.iterrows():
        t = rows["Text"][:100]
        u = rows["URL"]
        
        if t in texts:
            duplicates.append(t)
            duplicated_urls.append(u)
            # find index of duplicate in texts
            index = texts.index(t)
            # get url of duplicate
            url = texts_urls[index]
            # get label of duplicate
            label = labels[index]
            duplicated_labels.append(label)

            # print duplicate and url
            print("Duplicate: " + t)
            print("URL: " + u)
            print("Duplicate URL: " + url)
            print("")
            
        else:
            texts.append(t)
            texts_urls.append(u)
            labels.append(rows["Label"])
    print("Number of duplicates: " + str(len(duplicates)))
    """






