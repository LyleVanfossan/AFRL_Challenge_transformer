# AFRL_Challenge_transformer
Training Char level transformer models for predicting Aircraft Challenge Data

In this project we will be training a Transformer based neural network on Aircraft Challenge Data 10K this applicaiton trains on AFRC data, 
saves the predicted probability determined during new data generation, and processes the arithmetic encoding and decoding using those generated probabilities.

The initial process involved processing the seqenced data to better fit the transformer network, tokenizing the sequences along the way, training the network to the newly generated data, and using the
trained network to predict test data to create a predicted probabilities matrix. The examples below show the training sequence and outcome of training (shortened for brevity).
```bash
| epoch   1 |    50/ 1570 batches | lr 5.00 | ms/batch 46.11 | loss  5.17 | ppl   176.57
| epoch   1 |   100/ 1570 batches | lr 5.00 | ms/batch  7.36 | loss  3.85 | ppl    46.95
| epoch   1 |   150/ 1570 batches | lr 5.00 | ms/batch  7.39 | loss  3.17 | ppl    23.72
| epoch   1 |   200/ 1570 batches | lr 5.00 | ms/batch  7.99 | loss  2.77 | ppl    15.93
| epoch   1 |   250/ 1570 batches | lr 5.00 | ms/batch  8.04 | loss  2.42 | ppl    11.30
| epoch   1 |   300/ 1570 batches | lr 5.00 | ms/batch 11.32 | loss  2.51 | ppl    12.27
| epoch   1 |   350/ 1570 batches | lr 5.00 | ms/batch  7.90 | loss  2.53 | ppl    12.57
| epoch   1 |   400/ 1570 batches | lr 5.00 | ms/batch  7.80 | loss  2.48 | ppl    11.97
| epoch   1 |   450/ 1570 batches | lr 5.00 | ms/batch  7.68 | loss  2.33 | ppl    10.27
| epoch   1 |   500/ 1570 batches | lr 5.00 | ms/batch  7.65 | loss  2.32 | ppl    10.13
| epoch   1 |   550/ 1570 batches | lr 5.00 | ms/batch  7.37 | loss  2.29 | ppl     9.86
| epoch   1 |   600/ 1570 batches | lr 5.00 | ms/batch  7.24 | loss  2.28 | ppl     9.80
| epoch   1 |   650/ 1570 batches | lr 5.00 | ms/batch  8.14 | loss  2.27 | ppl     9.66
| epoch   1 |   700/ 1570 batches | lr 5.00 | ms/batch  9.93 | loss  2.25 | ppl     9.50
| epoch   1 |   750/ 1570 batches | lr 5.00 | ms/batch  8.56 | loss  2.25 | ppl     9.52
| epoch   1 |   800/ 1570 batches | lr 5.00 | ms/batch  7.94 | loss  2.26 | ppl     9.58
| epoch   1 |   850/ 1570 batches | lr 5.00 | ms/batch  7.81 | loss  2.24 | ppl     9.42
| epoch   1 |   900/ 1570 batches | lr 5.00 | ms/batch  7.84 | loss  2.23 | ppl     9.27
| epoch   1 |   950/ 1570 batches | lr 5.00 | ms/batch  8.16 | loss  2.22 | ppl     9.23
| epoch   1 |  1000/ 1570 batches | lr 5.00 | ms/batch  8.33 | loss  2.24 | ppl     9.38
| epoch   1 |  1050/ 1570 batches | lr 5.00 | ms/batch  7.88 | loss  2.23 | ppl     9.28
| epoch   1 |  1100/ 1570 batches | lr 5.00 | ms/batch  7.42 | loss  2.21 | ppl     9.16
| epoch   1 |  1150/ 1570 batches | lr 5.00 | ms/batch  8.26 | loss  2.22 | ppl     9.22
| epoch   1 |  1200/ 1570 batches | lr 5.00 | ms/batch  8.68 | loss  2.23 | ppl     9.28
| epoch   1 |  1250/ 1570 batches | lr 5.00 | ms/batch  7.64 | loss  2.23 | ppl     9.31
| epoch   1 |  1300/ 1570 batches | lr 5.00 | ms/batch  7.80 | loss  2.21 | ppl     9.12
| epoch   1 |  1350/ 1570 batches | lr 5.00 | ms/batch  7.58 | loss  2.22 | ppl     9.20
| epoch   1 |  1400/ 1570 batches | lr 5.00 | ms/batch  7.95 | loss  2.22 | ppl     9.17
| epoch   1 |  1450/ 1570 batches | lr 5.00 | ms/batch  7.50 | loss  2.20 | ppl     9.05
| epoch   1 |  1500/ 1570 batches | lr 5.00 | ms/batch  7.63 | loss  2.19 | ppl     8.94
| epoch   1 |  1550/ 1570 batches | lr 5.00 | ms/batch  7.85 | loss  2.22 | ppl     9.17
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 14.85s | valid loss  3.03 | valid ppl    20.61
-----------------------------------------------------------------------------------------
...
-----------------------------------------------------------------------------------------
| end of epoch  19 | time: 12.45s | valid loss  4.06 | valid ppl    58.17
-----------------------------------------------------------------------------------------
| epoch  20 |    50/ 1570 batches | lr 0.23 | ms/batch  8.35 | loss  1.42 | ppl     4.12
| epoch  20 |   100/ 1570 batches | lr 0.23 | ms/batch  7.65 | loss  1.38 | ppl     3.99
| epoch  20 |   150/ 1570 batches | lr 0.23 | ms/batch  7.58 | loss  1.39 | ppl     4.00
| epoch  20 |   200/ 1570 batches | lr 0.23 | ms/batch  7.57 | loss  1.37 | ppl     3.94
| epoch  20 |   250/ 1570 batches | lr 0.23 | ms/batch  7.63 | loss  1.37 | ppl     3.94
| epoch  20 |   300/ 1570 batches | lr 0.23 | ms/batch  7.62 | loss  1.38 | ppl     3.96
| epoch  20 |   350/ 1570 batches | lr 0.23 | ms/batch  7.64 | loss  1.39 | ppl     4.02
| epoch  20 |   400/ 1570 batches | lr 0.23 | ms/batch  7.62 | loss  1.36 | ppl     3.89
| epoch  20 |   450/ 1570 batches | lr 0.23 | ms/batch  7.79 | loss  1.37 | ppl     3.93
| epoch  20 |   500/ 1570 batches | lr 0.23 | ms/batch  7.66 | loss  1.36 | ppl     3.91
| epoch  20 |   550/ 1570 batches | lr 0.23 | ms/batch  7.63 | loss  1.36 | ppl     3.89
| epoch  20 |   600/ 1570 batches | lr 0.23 | ms/batch  7.78 | loss  1.38 | ppl     3.98
| epoch  20 |   650/ 1570 batches | lr 0.23 | ms/batch  7.64 | loss  1.39 | ppl     4.01
| epoch  20 |   700/ 1570 batches | lr 0.23 | ms/batch  7.66 | loss  1.38 | ppl     3.96
| epoch  20 |   750/ 1570 batches | lr 0.23 | ms/batch  7.67 | loss  1.38 | ppl     3.97
| epoch  20 |   800/ 1570 batches | lr 0.23 | ms/batch  7.69 | loss  1.38 | ppl     3.97
| epoch  20 |   850/ 1570 batches | lr 0.23 | ms/batch  7.59 | loss  1.37 | ppl     3.95
| epoch  20 |   900/ 1570 batches | lr 0.23 | ms/batch  7.65 | loss  1.38 | ppl     3.96
| epoch  20 |   950/ 1570 batches | lr 0.23 | ms/batch  7.73 | loss  1.35 | ppl     3.87
| epoch  20 |  1000/ 1570 batches | lr 0.23 | ms/batch  8.17 | loss  1.37 | ppl     3.94
| epoch  20 |  1050/ 1570 batches | lr 0.23 | ms/batch  7.58 | loss  1.37 | ppl     3.92
| epoch  20 |  1100/ 1570 batches | lr 0.23 | ms/batch  7.64 | loss  1.36 | ppl     3.91
| epoch  20 |  1150/ 1570 batches | lr 0.23 | ms/batch  7.76 | loss  1.36 | ppl     3.91
| epoch  20 |  1200/ 1570 batches | lr 0.23 | ms/batch  7.58 | loss  1.36 | ppl     3.91
| epoch  20 |  1250/ 1570 batches | lr 0.23 | ms/batch  7.73 | loss  1.37 | ppl     3.92
| epoch  20 |  1300/ 1570 batches | lr 0.23 | ms/batch  7.67 | loss  1.37 | ppl     3.94
| epoch  20 |  1350/ 1570 batches | lr 0.23 | ms/batch  7.65 | loss  1.36 | ppl     3.91
| epoch  20 |  1400/ 1570 batches | lr 0.23 | ms/batch  7.66 | loss  1.37 | ppl     3.94
| epoch  20 |  1450/ 1570 batches | lr 0.23 | ms/batch  7.64 | loss  1.37 | ppl     3.93
| epoch  20 |  1500/ 1570 batches | lr 0.23 | ms/batch  7.64 | loss  1.37 | ppl     3.93
| epoch  20 |  1550/ 1570 batches | lr 0.23 | ms/batch  7.66 | loss  1.38 | ppl     3.98
-----------------------------------------------------------------------------------------
| end of epoch  20 | time: 12.45s | valid loss  4.06 | valid ppl    57.99
-----------------------------------------------------------------------------------------
accuracy:  0.159782304942446
```

Additionally, some charts depicting loss and perplexity functions over 20 epochs.

## Loss
![image](https://github.com/LyleVanfossan/AFRL_Challenge_transformer/assets/60833582/5b0fb977-5412-4cdd-aff1-87d518788dba)

## Perplexity
![image](https://github.com/LyleVanfossan/AFRL_Challenge_transformer/assets/60833582/2a810e76-e164-418c-899b-a2d85f3c995d)

## Positional encoding
Standard positional encoding algorithm using sine and cosine functions.
```python
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
```

From there, we can use the generated prediciton matrix to process our training data trough an arithmetic encoding algorithm.

| Metric             | Value        |
|--------------------|--------------|
| Original filesize  | 139305 Bytes |
| Compressed filesize| 74811 Bytes  |
| Prediction Accuracy| 15.98 %      |
| Compression Ratio  | 186 %        |
| Decoding Accuracy  | 8.27 %       |
| Encoding time      | 3.36 seconds |
| Decoding time      | 12.01 seconds|
