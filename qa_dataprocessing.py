import pickle
import numpy as np

def pickle_raw_data():
    dict = {}
    raw_data = open('data/texts_raw_fixed.txt', 'r')
    i = 0
    for line in raw_data:
        print i
        line_arr = line.split('\t')
        dict[int(line_arr[0])] = [line_arr[1], line_arr[2].rstrip()]
        i += 1
    pickle_out = open("data/corpus_data.pickle","wb")
    pickle.dump(dict, pickle_out)
    pickle_out.close()

def pickle_training_data():
    arr = []
    training_data = open('data/train_random.txt')
    i = 0
    for line in training_data:
        print i
        line_arr = line.split('\t')
        arr.append([int(line_arr[0]), map(int, line_arr[1].split()), map(int, line_arr[2].split()[:20])])
        i += 1
    pickle_out = open("data/training_data.pickle","wb")
    pickle.dump(arr, pickle_out)
    pickle_out.close()

def pickle_dev_data():
        arr = []
        training_data = open('data/train_random.txt')
        i = 0
        for line in training_data:
            print i
            line_arr = line.split('\t')
            arr.append([int(line_arr[0]), map(int, line_arr[1].split()), map(int, line_arr[2].split())])
            i += 1
        pickle_out = open("data/dev_data.pickle","wb")
        pickle.dump(arr, pickle_out)
        pickle_out.close()

# pickle_training_data()
# pickle_raw_data()
# pickle_dev_data()
