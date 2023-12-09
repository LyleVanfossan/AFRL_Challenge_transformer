import math
import time
import os
import numpy as np
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment="AFC_cuda_transformer")
bptt = 50  # sequence length


class CharTransformer(nn.Module):

    def __init__(self, ntokens: int, embedding_dim: int, nhead: int,
                 hidden_dim: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_dim, nhead, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntokens, embedding_dim)
        self.hidden_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, ntokens)

        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
                Args:
                    src: Tensor, shape [seq_len, batch_size]
                    src_mask: Tensor, shape [seq_len, seq_len]

                Returns:
                    output Tensor of shape [seq_len, batch_size, ntoken]
                """

        src = self.embedding(src) * math.sqrt(self.hidden_dim)
        src = self.pos_encoder(src)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def batchify(data: Tensor, bsz: int) -> Tensor:
    seq_len = data.size(0) // bsz
    batch_data = data[:seq_len * bsz]
    batch_data = batch_data.view(bsz, seq_len).t().contiguous()
    return batch_data.to(device)

def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 50
    start_time = time.time()

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        output = model(data.cuda())
        output_flat = output.view(-1, ntokens)
        loss = criterion(output_flat, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        writer.add_scalar('Loss/train', loss, (65950 - 1)*(epoch-1)+i)
        writer.add_scalar('Perplexity/train', math.exp(loss), (65950 - 1)*(epoch-1)+i)
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def save():
    save_filename = os.path.dirname(os.path.realpath(filename)) + '/AFC_cuda_transformer_model_10k.ckpt'
    torch.save(model, save_filename)
    print('Saved model as %s' % save_filename)

def save_dataset(data, file_name):
    with open(f'{file_name}.txt', 'w') as f:
        for line in data:
            f.write(f'{line}\n')
        f.close()


if __name__ == '__main__':
    filename = "./data/Aircraft_Challenge_Data_10k_rows.txt"
    try:
        with open(filename, "r") as file:
            data = list(file.read())
            train_set, val_set, test_set = torch.utils.data.random_split(data, [0.8, 0.05, 0.15])
            test_set.indices.sort()
            val_set.indices.sort()
            train_set.indices.sort()
            # save_dataset(train_set, "train_set")
            # save_dataset(val_set, "val_set")
            # save_dataset(test_set, "test_set")
    except FileNotFoundError:
        print("File not found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unique characters
    chars = sorted(list(set(train_set)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda xx: [stoi[x] for x in xx]

    def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        return torch.tensor(encode(raw_text_iter), dtype=torch.long)



    ntokens = len(chars)  # size of vocabulary
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 3  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 2  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability
    model = CharTransformer(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    train_data = data_process(train_set)
    val_data = data_process(val_set)
    test_data = data_process(test_set)
    criterion = nn.CrossEntropyLoss()
    lr = 2.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.85)

    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)

    best_val_loss = float('inf')
    epochs = 5
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
        log_pt = 0
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train(model)
            val_loss = evaluate(model, val_data)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                  f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            scheduler.step()
        model.load_state_dict(torch.load(best_model_params_path))  # load best model states


    def char_tensor(string):
        tensor = torch.tensor(string).long()
        return tensor

    with torch.no_grad():
        data_tensors = data_process(test_set)
        file_len = len(data_tensors)
        prime_input = Variable(char_tensor(data_tensors).unsqueeze(0))

        predicted_probability = np.zeros((len(data_tensors), ntokens))
        for p in range(len(data_tensors)):
            out = torch.flatten(model(prime_input[:, p].cuda()))
            predicted_probability[p] = F.softmax(out).data.cpu().numpy()

        prob = np.zeros((file_len, ntokens))
        prob[1:] = predicted_probability[:file_len-1]
        np.save("MERL_prob.npy", prob)
        target = char_tensor(data_tensors[1:])
        correct_pred = np.equal(np.argmax(predicted_probability[1:], 1), target)
        accuracy = np.sum(correct_pred.numpy().astype(float)) / (len(data_tensors) - 1)

        print("accuracy: ", accuracy)
    save()
    writer.close()