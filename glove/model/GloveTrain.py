import os
os.chdir("C:\\Users\\wry\\Desktop\\deep-learning-examples-master\glove\\model")
import torch
import torch.utils.data as tud  #Pytorch读取训练集需要用到torch.utils.data类
from collections import Counter
from sklearn.metrics.pairwise import  cosine_similarity
import pandas as pd
import numpy as np
import scipy
import time
import math
import time
import sys
sys.path.append("C:\\Users\\wry\\Desktop\\deep-learning-examples-master\glove\\model")
import matplotlib.pyplot as plt
from glove_build_cooccurance import buildCooccuranceMatrix

from WordEmbeddingDataset import WordEmbeddingDataset
from GloveModel import GloveModel

EMBEDDING_SIZE = 50
MAX_VOCAB_SIZE = 5000
WINDOW_SIZE = 5

NUM_EPOCHS = 5
BATCH_SIZE = 1000
LEARNING_RATE = 0.05

TEXT_SIZE = 20000000
LOG_FILE = "../logs/glove-{}.log".format(EMBEDDING_SIZE)
WEIGHT_FILE = "../weights/glove-{}.th".format(EMBEDDING_SIZE)

def getCorpus(filetype, size):
    if filetype == 'dev':
        filepath = '../corpus/text8.dev.txt'
    elif filetype == 'test':
        filepath = '../corpus/text8.test.txt'
    else:
        filepath = '../corpus/text8.train.txt'

    with open(filepath, "r") as f:
        text = f.read()
        text = text.lower().split()
        text = text[: min(len(text), size)]
        vocab_dict = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1)) # 语料库中常用词和对应次数，非常用词用'<unk>'代替
        vocab_dict['<unk>'] = len(text) - sum(list(vocab_dict.values()))
        idx_to_word = list(vocab_dict.keys()) # 常用词（无重复词）
        word_to_idx = {word:ind for ind, word in enumerate(idx_to_word)} # 常用词对应索引
        word_counts = np.array(list(vocab_dict.values()), dtype=np.float32)
        word_freqs = word_counts / sum(word_counts)
        print("Words list length:{}".format(len(text)))
        print("Vocab size:{}".format(len(idx_to_word)))
    return text, idx_to_word, word_to_idx, word_counts, word_freqs

# def buildCooccuranceMatrix(text, word_to_idx,WINDOW_SIZE):
#     vocab_size = len(word_to_idx)
#     maxlength = len(text)
#     text_ids = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in text]
#     cooccurance_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
#     # print("Co-Matrix consumed mem:%.2fMB" % (sys.getsizeof(cooccurance_matrix)/(1024*1024)))
#     for i, center_word_id in enumerate(text_ids):
#         window_indices = list(range(i - WINDOW_SIZE, i)) + list(range(i + 1, i + WINDOW_SIZE + 1))
#         window_indices = [i % maxlength  for i in window_indices if i>=0 and i<=maxlength]
#         window_word_ids = [text_ids[index]  for index in window_indices ]
#         for context_word_id in window_word_ids:
#             cooccurance_matrix[center_word_id][context_word_id] += 1
#         if (i+1) % 1000000 == 0:
#             print(">>>>> Process %dth word" % (i+1))
#     print(">>>>> Build co-occurance matrix completed.")
#     return cooccurance_matrix


def buildWeightMatrix_apply(element,xmax):
    return math.pow(element/xmax,0.75) if element<xmax else 1.0


def buildWeightMatrix(co_matrix):
    xmax = 100.0
    # weight_matrix = np.zeros_like(co_matrix, dtype=np.float32)
    # print("Weight-Matrix consumed mem:%.2fMB" % (sys.getsizeof(weight_matrix) / (1024 * 1024)))
    # for i in range(co_matrix.shape[0]):
    #     for j in range(co_matrix.shape[1]):
    #         weight_matrix[i][j] = math.pow(co_matrix[i][j] / xmax, 0.75) if co_matrix[i][j] < xmax else 1
    #     if (i+1) % 1000 == 0:
    #         print(">>>>> Process %dth weight" % (i+1))
    buildWeightMatrix_apply_np = np.vectorize(buildWeightMatrix_apply)
    weight_matrix = buildWeightMatrix_apply_np(co_matrix,xmax)
    print("Weight-Matrix consumed mem:%.2fMB" % (sys.getsizeof(weight_matrix) / (1024 * 1024)))
    print(">>>>> Build weight matrix completed.")
    return weight_matrix

def find_nearest(word, embedding_weights):
    index = word_to_idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:10]]

def asMinutes(s):
    h = math.floor(s / 3600)
    s = s - h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return '%dh %dm %ds' % (h, m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent # 估计总时间
    rs = es - s # 剩下需要的时间
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def loadModel():
    path = WEIGHT_FILE
    model = GloveModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
    model.load_state_dict(torch.load(path))
    return model

def findRelationshipVector(word1, word2, word3):
    word1_idx = word_to_idx[word1]
    word2_idx = word_to_idx[word2]
    word3_idx = word_to_idx[word3]
    embedding = glove_matrix[word2_idx] - glove_matrix[word1_idx] + glove_matrix[word3_idx]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in glove_matrix])
    for i in cos_dis.argsort()[:5]:
        print("{} to {} as {} to {}".format(word1, word2, word3, idx_to_word[i]))

if __name__ == '__main__':
    text, idx_to_word, word_to_idx, word_counts, word_freqs = getCorpus('train', size=TEXT_SIZE)    #加载语料及预处理
    co_matrix = buildCooccuranceMatrix(text, word_to_idx,WINDOW_SIZE)    #构建共现矩阵
    weight_matrix = buildWeightMatrix(co_matrix)             #构建权重矩阵
    dataset = WordEmbeddingDataset(co_matrix, weight_matrix) #创建dataset
    dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model = GloveModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE) #创建模型
    #model = loadModel()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE) #选择Adagrad优化器


    print_every = 10000
    save_every = 50000
    epochs = NUM_EPOCHS # 10,10轮迭代
    iters_per_epoch = int(dataset.__len__() / BATCH_SIZE) # 每轮迭代计算次数，BATCH_SIZE：每次计算样本数
    total_iterations = iters_per_epoch * epochs # 总计算次数
    print("Iterations: %d per one epoch, Total iterations: %d " % (iters_per_epoch, total_iterations))

    start = time.time()
    for epoch in range(epochs): # 第几轮迭代
        loss_print_avg = 0
        iteration = iters_per_epoch * epoch # 第几次计算
        for i, j, co_occur, weight in dataloader:
            iteration += 1 #计算次数+1
            optimizer.zero_grad()   #每一批样本训练前重置缓存的梯度
            loss = model(i, j, co_occur, weight)    #前向传播
            loss.backward()     #反向传播
            optimizer.step()    #更新梯度
            loss_print_avg += loss.item()

            if iteration % print_every == 0:
                time_desc = timeSince(start, iteration / total_iterations)
                iter_percent = iteration / total_iterations * 100 # 计算进度百分比
                loss_avg = loss_print_avg / print_every # 每次计算的平均损失
                loss_print_avg = 0
                with open(LOG_FILE, "a") as fout:
                    fout.write("epoch: %d, iter: %d (%.4f%%), loss: %.5f, %s\n" %
                               (epoch, iteration, iter_percent, loss_avg, time_desc))
                print("epoch: %d, iter: %d/%d (%.4f%%), loss: %.5f, %s" %
                      (epoch, iteration, total_iterations, iter_percent, loss_avg, time_desc))
            if iteration % save_every == 0:
                torch.save(model.state_dict(), WEIGHT_FILE)
    torch.save(model.state_dict(), WEIGHT_FILE)

    glove_matrix = model.gloveMatrix()
    for word in ["good", "one", "green", "like", "america", "queen", "better", "paris", "work", "computer", "language"]:
        print(word, find_nearest(word, glove_matrix))
    findRelationshipVector('man', 'king', 'woman')
    findRelationshipVector('america', 'washington', 'france')
    findRelationshipVector('good', 'better', 'little')

    #数据降维以及可视化
    candidate_words = ['one','two','three','four','five','six','seven','eight','night','ten','color','green','blue','red','black',
                       'man','woman','king','queen','wife','son','daughter','brown','zero','computer','hardware','software','system','program',
                       'america','china','france','washington','good','better','bad']
    candidate_indexes = [word_to_idx[word] for word in candidate_words]
    choosen_indexes = candidate_indexes
    choosen_vectors = [glove_matrix[index] for index in choosen_indexes]

    U, S, VH = np.linalg.svd(choosen_vectors, full_matrices=False)
    for i in range(len(choosen_indexes)):
        plt.text(U[i, 0], U[i, 1], idx_to_word[choosen_indexes[i]])

    coordinate = U[:, 0:2]
    plt.xlim((np.min(coordinate[:, 0]) - 0.1, np.max(coordinate[:, 0]) + 0.1))
    plt.ylim((np.min(coordinate[:, 1]) - 0.1, np.max(coordinate[:, 1]) + 0.1))
    plt.show()