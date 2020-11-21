import pandas as pd
import numpy as np

# Import Mathplotlib #
import matplotlib
matplotlib.use('TkAgg')

# Library For - Operating System #
import os

# Library For - Neural Networks #
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
from collections import Counter

# Clean The Console #
clear = lambda: os.system('cls')
clear()

# Reviews = All The Sentence's #
reviews = pd.read_csv('reviews.txt', header = None)

# Labels = Negetive / Positive #
labels = pd.read_csv('labels.txt', header = None)

# Dictionary = Count The Number of Performence of each Word #
total_counts = Counter()
Count = 0
clear()
print(" =========================")
print(" ===== Deep Learning =====")
print(" =========================")
for _, row in reviews.iterrows():
    if Count == 10:
        Index = 0
        while Index < 30:
            print("", end = "\b", flush = True)
            Index = Index + 1
            pass
        Count = 0
    else:
        print(" * ", end = "")
        Count = Count + 1
        pass
    for word in row[0].split(' '):
        total_counts[word] += 1

# Sorting The 'total_counts' According to the Number of Performance #
vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]

# Word2Index = [Value , Index In The Array Of Vocabulary] #
word2index = {}
for index, word in enumerate(vocab):
    word2index[word] = index

# This Function Make Text To Vector according to the Performance in Vocabulary #
def text_to_vector(text): 
    word_vector = np.zeros(len(vocab))
    for word in text.split(' '):
        if word in vocab:
            word_vector[word2index[word]] = 1
    return word_vector

# len(List) = 25000 , len(List[0]) = 10000 , ... , len(List[25000]) = 10000 #
print()
print()
Count = 0                   # Count For Printing #
Average_One = 0             # Average For Positive #
Average_Zero = 0            # Average For Negetive #
Count_Sentence_Pos = 0      # Count Sentence Positive #
Count_Sentence_Neg = 0      # Count Sentence Negetive #
Flag = False
word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_) 
for ii, (_, text) in enumerate(reviews.iterrows()):
    if Count == 10:
        Index = 0
        while Index < 30:
            print("", end = "\b", flush = True)
            Index = Index + 1
            pass
        Count = 0
    else:
        print(" * ", end = "")
        Count = Count + 1
        pass
    word_vectors[ii] = text_to_vector(text[0])

    if Flag == False:
        # Check If File's Exist #
        if os.path.exists(os.getcwd() + "\\" + "Final_File_Positive.txt"):
            os.remove(os.getcwd() + "\\" + "Final_File_Positive.txt")
            pass
        if os.path.exists(os.getcwd() + "\\" + "Final_File_Negetive.txt"):
            os.remove(os.getcwd() + "\\" + "Final_File_Negetive.txt")
            pass
        Flag = True
        pass
    
    # Not In Used #
    '''
    Count_One = word_vectors[ii].tolist().count(1)
    Count_Zero = word_vectors[ii].tolist().count(0)
    '''

    if ii % 3 == 0:
        with open(os.getcwd() + "\\" + "Final_File_Positive.txt",'a',encoding='utf-8') as Final_File:
            Final_File.write(str(Count_Sentence_Pos) + ' ===> ' + str(text[0]) + '\n\n')
            Final_File.write('=======================================================================================================\n\n')
            Count_Sentence_Pos = Count_Sentence_Pos + 1
            pass
        Average_One = Average_One + 1
        pass
    else:
        with open(os.getcwd() + "\\" + "Final_File_Negetive.txt",'a',encoding='utf-8') as Final_File:
            Final_File.write(str(Count_Sentence_Neg) + ' ===> ' + str(text[0]) + '\n\n')
            Final_File.write('=======================================================================================================\n\n')
            Count_Sentence_Neg = Count_Sentence_Neg + 1
            pass
        Average_Zero = Average_Zero + 1
        pass

# Print The Final Average For - Positive File #
if os.path.exists(os.getcwd() + "\\" + "Final_File_Positive.txt"):
    with open(os.getcwd() + "\\" + "Final_File_Positive.txt",'a',encoding='utf-8') as Final_File:
        Final_File.write('\n')
        Final_File.write('\n')
        Positive = str(float(Average_One/len(word_vectors)))
        Final_File.write('Average Of Positive Is ===> ' + Positive + '%')
        pass
    pass 

# Print The Final Average For - Negetive File #
if os.path.exists(os.getcwd() + "\\" + "Final_File_Negetive.txt"):
    with open(os.getcwd() + "\\" + "Final_File_Negetive.txt",'a',encoding='utf-8') as Final_File:
        Final_File.write('\n')
        Final_File.write('\n')
        Negetive = str(float(Average_Zero/len(word_vectors)))
        Final_File.write('Average Of Negetive Is ===> ' + Negetive + '%')
        pass
    pass 

Y = (labels == 'positive').astype(np.int_)
records = len(labels)

shuffle = np.arange(records)
np.random.shuffle(shuffle)

# 90% For Testing , And 10% For Training #
test_fraction = 0.9 

# Divide For Training And Testing #
train_split, test_split = shuffle[:int(records * test_fraction)], shuffle[int(records * test_fraction):] 
trainX, trainY = word_vectors[train_split, :], to_categorical(Y.values[train_split].T[0], 2)
testX, testY = word_vectors[test_split, :], to_categorical(Y.values[test_split].T[0], 2)

# Build The Model #
def build_model():
    net = tflearn.input_data([None, 10000])
    net = tflearn.fully_connected(net, 200, activation='ReLU')
    net = tflearn.fully_connected(net, 25, activation='ReLU')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')

    # Deep Neural Network #
    model = tflearn.DNN(net) 
    return model

# Not In Used #
'''
# This Function Creating The Final File For ===> Result #
def Create_Final_File():
    Average_One = 0
    Average_Zero = 0

    # Check If File Exist #
    if os.path.exists(os.getcwd() + "\\" + "Final_File.txt"):
        os.remove(os.getcwd() + "\\" + "Final_File.txt")

    # Open New File For Writing #
    with open(os.getcwd() + "\\" + "Final_File.txt",'w',encoding='utf-8') as Final_File:
        Index_One = 0
        while Index_One < len(word_vectors):
            Count_One = word_vectors[Index_One].tolist().count(1)
            Count_Zero = word_vectors[Index_One].tolist().count(0)
            if Count_One > Count_Zero:
                Final_File.write(str(text[0]) + ' ===> Positive\n')
                Average_One = Average_One + 1
            else:
                Final_File.write(str(text[0]) + ' ===> Negetive\n')
                Average_Zero = Average_Zero + 1
                pass
            Index_One = Index_One + 1
            pass

        Final_File.write('\n')
        Final_File.write('\n')
        Positive = str(float(Average_One/len(word_vectors)))
        Negetive = str(float(Average_Zero/len(word_vectors)))
        Final_File.write('Average Of Positive Is ===> ' + Positive + '%')
        Final_File.write('Average Of Negetive Is ===> ' + Negetive + '%')
        pass
    pass

# Create Final File #
Create_Final_File()
'''

# TrainX = First Part Of Training , TrainY = Second Part Of Training, Validation == Development / Prediction, Show Metrics = Show The Result's , n_epoch = Divide The File To 30 Parts #
model = build_model()
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=30) 

# Make The Prediction For The Model , How Much The Model Succeed #
predictions = (np.array(model.predict(testX))[:, 0] >= 0.5).astype(np.int_) 

# Check How Much The Model Is Accurate #
test_accuracy = np.mean(predictions == testY[:, 0], axis=0)                 
print("Test accuracy: ", test_accuracy)





