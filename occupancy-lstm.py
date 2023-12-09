import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import random
import time
import math

from tqdm import tqdm
from data_builder import data_builder
from torch.utils import data as dataloader
import torch.nn.functional as F

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, batch_size=100, time_step=25, drop_prob=0.5, lr=0.01):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.time_step = time_step
        self.drop_prob = drop_prob
        self.lr = lr

        self.dropout = nn.Dropout(drop_prob)
        self.encoder = nn.Embedding(self.input_size, self.input_size)
        self.rnn = nn.LSTM(self.input_size,
                           self.hidden_size,
                           self.n_layers,
                           dropout=self.drop_prob,
                           batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))


def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len - 1)
        end_index = start_index + chunk_len + 1
        chunk = data_split[start_index: end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target


def training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        chunk = data_split[bi * chunk_len: (bi * chunk_len) + chunk_len + 1]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target


def train(inp, target):
    hidden = model.init_hidden(batch_size)
    if cuda:
        if isinstance(hidden, tuple):
            hidden = (hidden[0].cuda(), hidden[1].cuda())
    model.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = model(inp[:,c], hidden)
        loss += criterion(output.view(batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / chunk_len


def char_tensor(string):
    tensor = torch.tensor(string).long()
    return tensor


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def save():
    save_filename = os.path.dirname(os.path.realpath(filename)) + '/AFC_cuda_model_1_50_l2_dr_0_1_e_1000_2.ckpt'
    torch.save(model, save_filename)
    print('Saved model as %s' % save_filename)


def save_dataset(data, file_name):
    with open(f'{file_name}.txt', 'w') as f:
        for line in data:
            f.write(f'{line}\n')
        f.close()

# Opening the file and reading the text
filename = "./data/Aircraft_Challenge_Data_10k_rows.txt"

if __name__ == '__main__':
    try:
        with open(filename, 'r') as file:
            data = file.read()
            data = data.split('\n')
            train_set, val_set = torch.utils.data.random_split(data, [9000, 1000])
            save_dataset(train_set, 'train_set')
            save_dataset(val_set, 'val_set')
            file.close()
    except OSError:
        print("File not found")

    cuda = False
    if torch.cuda.is_available():
        cuda = True

    # list all the used char
    vocab = {token: idx for idx, token in enumerate(set(list(data)))}
    data_tensors = [torch.tensor([vocab[token] for token in [*seq]], dtype=torch.long) for seq in data]
    file_len, vocab_size = len(data_tensors), len(vocab)
    print("Data has %d characters in total, and %d unique characters" % (file_len, vocab_size))

    # Defining model parameters
    input_size = vocab_size
    output_size = input_size
    hidden_size = 100
    num_layer = 3
    print("Model Parameter")
    print("input/output size: ", input_size)
    print("hidden size: ", hidden_size)
    print("LSTM layer: ", num_layer)

    # Defining training parameter
    num_epoch = 10
    batch_size = 1
    shuffle = False
    chunk_len = int(len(data_tensors)/batch_size)
    learning_rate = 0.01
    dropout_prob = 0.5
    print_every = 100

    print("Trainig Parameter")
    print("num of epoch: ", num_epoch)
    print("batch size: ", batch_size)
    print("chunk len: ", chunk_len)
    print("Total iteration(chunk len): ", chunk_len)
    print("learning rate: ", learning_rate)
    print("dropout probability: ", dropout_prob)

    # Define model
    model = CharRNN(input_size, hidden_size, output_size, num_layer, batch_size, drop_prob=dropout_prob)

    decoder_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    if cuda:
        print("Using cuda!\n")
        model.cuda()

    start = time.time()
    all_losses = []
    loss_avg = 0

    custom_dataset = data_builder(data_tensors)
    train_data = dataloader.DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6)
    train_loss = []
    try:
        print("Training for %d epochs..." % num_epoch)
        for epoch in tqdm(range(1, num_epoch + 1)):

            # hidden = model.init_hidden(batch_size)
            # if cuda:
            #     if isinstance(hidden, tuple):
            #         hidden = (hidden[0].cuda(), hidden[1].cuda())
            # model.zero_grad()
            loss = 0
            for inps_, target_ in train_data:
                if len(inps_) == batch_size:
                    if cuda:
                        inps = torch.argmax(inps_.cuda(), axis=1)
                        target = torch.argmax(target_.cuda(), axis=1)

                    output = model(inps)
                    output = output.view(batch_size, -1)
                    loss += criterion(output.view(batch_size, -1), target)

                    # print(loss)

            loss.backward()
            decoder_optimizer.step()

            train_loss.append(loss.item() / chunk_len)
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / num_epoch * 100, train_loss[-1]))

        print("Saving...")
        save()


    except KeyboardInterrupt:
        print("Saving before quit...")
        save()