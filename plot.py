import pandas as pd
import matplotlib.pyplot as plt
import ast
import math
import collections

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