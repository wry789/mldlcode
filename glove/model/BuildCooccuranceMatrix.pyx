import numpy as np
import sys
import time
cimport numpy as np

def buildCooccuranceMatrix(list text, dict word_to_idx,int WINDOW_SIZE):
    cdef int vocab_size ,maxlength,i,center_word_id,context_word_id
    cdef list text_ids,window_indices,window_word_ids
    cdef float start_time
    cdef np.ndarray[np.int32_t, ndim=2] cooccurance_matrix
    vocab_size = len(word_to_idx)
    maxlength = len(text)
    text_ids = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in text]
    cooccurance_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    print("Co-Matrix consumed mem:%.2fMB" % (sys.getsizeof(cooccurance_matrix)/(1024*1024)))
    for i, center_word_id in enumerate(text_ids):
        window_indices = list(range(i - WINDOW_SIZE, i)) + list(range(i + 1, i + WINDOW_SIZE + 1))
        window_indices = [i % maxlength  for i in window_indices if i>=0 and i<=maxlength]
        window_word_ids = [text_ids[index]  for index in window_indices ]
        for context_word_id in window_word_ids:
            cooccurance_matrix[center_word_id][context_word_id] += 1
        if (i+1) % 1000000 == 0:
            print(">>>>> Process %dth word" % (i+1))
    print(">>>>> Build co-occurance matrix completed.")

    return cooccurance_matrix