from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split

class Collater:
    def __init__(self, src_lang, trg_lang=None, predict=False):
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.predict = predict

    def __call__(self, batch):
        if self.predict:
            return nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=self.src_lang.PAD_idx
            )

        src_tensors, trg_tensors = zip(*batch)
        src_tensors = nn.utils.rnn.pad_sequence(
            src_tensors, batch_first=True, padding_value=self.src_lang.PAD_idx
        )
        trg_tensors = nn.utils.rnn.pad_sequence(
            trg_tensors, batch_first=True, padding_value=self.trg_lang.PAD_idx
        )
        return src_tensors, trg_tensors


def sentence_to_tensor(sentence, lang):
    indexes = [lang.word2index[w] for w in lang.sentence_to_words(sentence)]
    indexes = [lang.SOS_idx] + indexes + [lang.EOS_idx]
    return torch.LongTensor(indexes)


def pairs_to_tensors(pairs, src_lang, trg_lang):
    tensors = [
        (sentence_to_tensor(src, src_lang), sentence_to_tensor(trg, trg_lang))
        for src, trg in tqdm(pairs, desc="creating tensors")
    ]
    return tensors

def score(true_expansion, pred_expansion):
    return int(true_expansion == pred_expansion)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataset(root_folder="./dataset", seed=1234, debug=False):
    data_path = os.path.join(root_folder, "dataset.txt")
    train_path, val_path, test_path = os.path.join(root_folder, "train.txt"), \
                                        os.path.join(root_folder, "val.txt"), \
                                        os.path.join(root_folder, "test.txt")
    data = np.loadtxt(data_path, dtype=str, delimiter="=")
    train, val_test = train_test_split(data, test_size = 0.2, random_state=seed)
    val, test = train_test_split(val_test, test_size = 0.5, random_state=seed)
    if debug:
        print(train.shape, val.shape, test.shape)
    np.savetxt(train_path, train, fmt="%s", delimiter='=')
    np.savetxt(val_path, val, fmt="%s", delimiter='=')
    np.savetxt(test_path, test, fmt="%s", delimiter='=')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
