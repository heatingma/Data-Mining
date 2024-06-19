import numpy as np
from sklearn.model_selection import train_test_split
import jieba
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator


data = np.load('../cache/movie_comments.npy')


labels = np.where(data[:, 4].astype(int) > 3, 1, 0)


texts = data[:, 2]
labels = labels.astype(int)

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

def yield_tokens(text_iter):
    for text in text_iter:
        yield list(jieba.cut(text))

vocab = build_vocab_from_iterator(yield_tokens(X_train), specials=["<unk>"], min_freq=1)


vocab.set_default_index(vocab["<unk>"])


def text_pipeline(x):
    return [vocab[token] for token in list(jieba.cut(x))]


def label_pipeline(x):
    return int(x)


class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = torch.tensor(text_pipeline(self.texts[idx]), dtype=torch.int64)
        label = torch.tensor(label_pipeline(self.labels[idx]), dtype=torch.float32)
        return text, label


def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        lengths.append(len(_text))
        text_list.append(_text)
    text_list = pad_sequence(text_list, batch_first=True)
    label_list = torch.tensor(label_list, dtype=torch.float32)
    return text_list, label_list

def get_data_loaders(batch_size=8):
    train_dataset = ReviewDataset(X_train, y_train)
    test_dataset = ReviewDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    return train_dataloader, test_dataloader, vocab