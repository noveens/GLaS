import numpy as np
from scipy.sparse import csr_matrix
from tqdm.notebook import tqdm
import torch
from torch.autograd import Variable
import random

class DataLoader():
    def __init__(self, train_features, train_labels):
        
        self.x_data, num_points, num_words = read_sparse(
            train_features, return_dict = True, padding = False, tf_idf = True
        )
        self.y_data, _, num_labels = read_sparse(
            train_labels, return_dict = True, padding = False, tf_idf = False
        )
        
        self.num_points = num_points
        self.num_labels = num_labels
        self.num_words = num_words
        
        # Unroll
        self.x, self.y, self.indices = [], [], []
        for x_index in range(len(self.x_data)):
            for label_index in range(len(self.y_data[x_index])):
                self.x.append(self.x_data[x_index])
                self.y.append(self.y_data[x_index][label_index])
                self.indices.append(x_index)
                
        # Make self.y_data a set
        for x_index in range(len(self.x_data)):
            self.y_data[x_index] = set(self.y_data[x_index])
                
        # Shuffle
        self.shuffle_stochastic()
        
    def shuffle_stochastic(self):
        indices = np.arange(len(self.x))
        np.random.shuffle(indices)
        self.x = np.array(self.x)[indices].tolist()
        self.y = np.array(self.y)[indices].tolist()
        self.indices = np.array(self.indices)[indices].tolist()
        
    def pad(self, arr, pad_index = -1):
        pad_len = max(map(len, arr))
        for i in range(len(arr)):
            while len(arr[i]) < pad_len: arr[i].append(pad_index)
        return arr
    
    def serialize(self, arr, drop_prob = None, tf_idf = False):
        words, offsets, tf = [], [], []
        for i in arr:
            offsets.append(len(words))
            
            # Dropout without padding idx
            if drop_prob is not None:
                probs = np.random.uniform(0, 1, len(i))
                i = np.array(i)
                i = i[probs >= drop_prob].tolist()
            
            words += [ ii[0] for ii in i ]
            tf += [ ii[1] for ii in i ]
            
        if tf_idf == True: return words, offsets, tf
        return words, offsets, None
        
    def iter(self, bsz, dropout = None, tf = False): 
        # Shuffling to get different negatives every time
        self.shuffle_stochastic()
        
        pbar = tqdm(range(0, len(self.x), bsz))
        for i in pbar:
            end_index = min(i + bsz, len(self.x))
            
            x, offsets, tf_idf = self.serialize(self.x[i:end_index], drop_prob = dropout, tf_idf = tf)
            y = self.y[i:end_index]

            all_ys = set(self.y[i:end_index])
            negatives = []
            for j in range(len(y)):
                negatives.append(list(all_ys - self.y_data[self.indices[i + j]]))
            negatives = self.pad(negatives, self.num_labels)
                
            if tf == True: tf_idf = Variable(torch.cuda.FloatTensor(tf_idf))
                
            # Yield
            yield Variable(torch.cuda.LongTensor(x)), \
                  Variable(torch.cuda.LongTensor(offsets)), \
                  tf_idf, \
                  Variable(torch.cuda.LongTensor(y)).unsqueeze(-1), \
                  Variable(torch.cuda.LongTensor(list(all_ys))), \
                  Variable(torch.cuda.LongTensor(negatives)), \
                  pbar
            
    def iter_eval(self, bsz, tf = False):
        pbar = tqdm(range(0, len(self.x_data), bsz))
        for i in pbar:
            end_index = min(i + bsz, len(self.x_data))
            x, offsets, tf_idf = self.serialize(self.x_data[i:end_index], tf_idf = tf)
            
            if tf == True: tf_idf = Variable(torch.cuda.FloatTensor(tf_idf))
            
            yield Variable(torch.cuda.LongTensor(x)), \
                  Variable(torch.cuda.LongTensor(offsets)), \
                  tf_idf, \
                  i, pbar

def read_sparse(file, return_dict = False, return_separate = False, padding = True, tf_idf = False):
    matrix = None
    nr, nc = None, None

    with open(file, "r") as f:
        line = f.readline()
        nr, nc = list(map(int, line.strip().split()))

        rows = []; cols = []; data = []

        at = 0; max_col = 0
        line = f.readline()
        while line:
            line = line.strip().split()
            for temp in line:
                index, val = int(temp.split(":")[0]), float(temp.split(":")[1])
                
                rows.append(at)
                cols.append(index)
                data.append(float(val))
                max_col = max(max_col, index)
            at += 1
            line = f.readline()

        if return_separate == True:
            matrix = [ 
                np.array(rows), 
                np.array(cols), 
                np.array(data) 
            ]
        
        elif return_dict == False:
            data = np.array(data)
            rows = np.array(rows)
            cols = np.array(cols)
            matrix = csr_matrix((data, (rows, cols)), shape = (nr, nc), dtype = np.bool_)
        
        elif return_dict == True:
            matrix = []
            for i in range(at): matrix.append([])
            
            for i in range(len(rows)):
                if tf_idf == False: matrix[rows[i]].append(cols[i])
                else: matrix[rows[i]].append([ cols[i], data[i] ])
                
            max_num = 0
            for i in range(len(matrix)):
                max_num = max(max_num, len(matrix[i]))
            print(max_num)
                
            if padding == True: 
                for i in range(len(matrix)):
                    while len(matrix[i]) != max_num:
                        matrix[i].append(nc)
                matrix = np.array(matrix)

    return matrix, nr, nc
