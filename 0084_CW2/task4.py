import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from BM25 import select_top_passages, load_document
from task2 import text_preprocess
import gensim.downloader as api

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # h0 = torch.zeros(1, x.size(0), self.hidden_dim).requires_grad_()
        # c0 = torch.zeros(1, x.size(0), self.hidden_dim).requires_grad_()
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze(1)

def embedding(file_path):
    model = api.load('glove-twitter-100')
    data_df = load_document(
        file_path, names=None
    )
    sentences = text_preprocess(data_df, save_name='train', do_subsample=True, task4=True)
    res_all = []
    for words in tqdm(sentences):
        res_sentence = []
        words_lst = words.split(',')
        [res_sentence.append(model[word]) for word in words_lst[:100] if word in model]
        if len(res_sentence) < 100:
            for i in range(100 - len(res_sentence)):
                res_sentence.append([0] * 100)

        res_all.append(res_sentence)
    res_all = np.asarray(res_all)
    print(res_all.shape)

    return np.asarray(data_df['qid'].values), np.asarray(data_df['pid'].values),np.asarray(data_df['relevancy'].values), res_all

if __name__ == "__main__":
    # construct the training, validation and testing datasets

    train_file_path = 'train_data.tsv'
    val_file_path = 'validation_data.tsv'
    # test_file_path = 'test-queries.tsv'
    Xy_train_qid, Xy_train_pid, Xy_train_label, Xy_train_data = embedding(train_file_path)
    Xy_val_qid, Xy_val_pid, Xy_val_label, Xy_val_data = embedding(val_file_path)

    # np.save('Xy_train_qid.npy', Xy_train_qid)
    # np.save('Xy_train_pid.npy', Xy_train_pid)
    # np.save('Xy_train_label.npy', Xy_train_label)
    # np.save('Xy_train_data.npy', Xy_train_data)
    # print('train data saved')
    # Xy_train_qid = np.load('Xy_train_qid.npy')
    # Xy_train_pid = np.load('Xy_train_pid.npy')
    # Xy_train_label = np.load('Xy_train_label.npy')
    # Xy_train_data = np.load('Xy_train_data.npy').astype(np.float64)

    # build the model
    num_epochs = 10
    batch_size = 5000
    model = LSTMModel(100, 128, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    # train the model
    for i in range(num_epochs):
        print('Epoch: {}'.format(i))
        for i in range(0, len(Xy_train_qid), batch_size):
            X_batch = torch.from_numpy(Xy_train_data[i:i+batch_size, :]).double()
            y_batch = torch.from_numpy(Xy_train_label[i:i+batch_size]).long()

            optimizer.zero_grad()
            y_pred = model(X_batch)

            loss = loss_function(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    # save the model
    np.save('y_test_pred.npy', y_pred.detach().numpy())