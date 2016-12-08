import numpy as np


def prepare_ques_batch(questions, ques_len, max_ques_len, embeddings, word_dim, ix_to_word):
    n_samples = len(questions)
    X_ques_batch = np.zeros((n_samples, max_ques_len, word_dim))
    
    for ques_ix, ques in enumerate(questions):
        for token_ix, token in enumerate(ques[:ques_len[ques_ix]]):
            token_emb = embeddings(ix_to_word[str(token)])
            X_ques_batch[ques_ix, token_ix, :] = token_emb.vector
    return X_ques_batch


def prepare_im_batch(fv_im, im_ix):
    X_im_batch = []
    for ix in im_ix:
        X_im_batch.append(fv_im[ix])
    return np.asarray(X_im_batch)

def get_batches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")
    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])
    return zip(range(len(minibatches)), minibatches)