
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Length of the text

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

    
if __name__ == "__main__":
    length_bins(5, "tweet")
