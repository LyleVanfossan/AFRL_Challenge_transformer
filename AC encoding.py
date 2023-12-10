import numpy as np
from decimal import Decimal
from decimal import getcontext
import time


start =time.time()
np.set_printoptions(threshold=np.inf)

def encoder(arr_prob,arr_data):
    low = 0.0
    high = 1.0
    Range = 1.0
    str = []
    for i in range(len(arr_data)):
        rangeLow = np.sum(arr_prob[i,:arr_data[i]])
        rangeHigh = np.sum(arr_prob[i,:arr_data[i]+1])
        high = Decimal(low) + Decimal(Range) * Decimal(rangeHigh)
        low = Decimal(low) + Decimal(Range) * Decimal(rangeLow)
        while(low>Decimal(1/2) or high <Decimal(1/2)):
            if low > Decimal(1/2):
                str.append(1)
                low = (Decimal(low) - Decimal(1/2))*Decimal(2)
                high = (Decimal(high) - Decimal(1/2))*Decimal(2)
            if high < Decimal(1/2):
                str.append(0)
                low = Decimal(low)*Decimal(2)
                high = Decimal(high) * Decimal(2)
        Range = Decimal(high) - Decimal(low)
    str.append(1)
    return(str)

def decoder(arr_prob,string):
    low = 0.0
    high = 1.0
    Range = 1.0
    result = []
    tag_length = 50
    next = tag_length
    for i in range(tag_length):
        string.append(0)


    tag = Decimal(0)
    for i, x in enumerate(string[:tag_length]):
        tag = Decimal(tag) + Decimal(2**(-i-1)*x)


    for i, prob_distribution in enumerate(arr_prob):
        for key, value in enumerate(prob_distribution):
            rangeLow = np.sum(prob_distribution[:key])
            rangeHigh = np.sum(prob_distribution[:key + 1])
            if(tag>=Decimal(low) +Decimal(Range)*Decimal(rangeLow) and tag<Decimal(low) +Decimal(Range)*Decimal(rangeHigh)):
                result.append(key)
                high = Decimal(low) + Decimal(Range) * Decimal(rangeHigh)
                low = Decimal(low) + Decimal(Range) * Decimal(rangeLow)
                Range = Decimal(high) - Decimal(low)
                break
        if i ==len(arr_prob)-1:
            break

        while (low > Decimal(1 / 2) or high < Decimal(1 / 2)):
            if low > Decimal(1 / 2):
                low = (Decimal(low) - Decimal(1 / 2)) * Decimal(2)
                high = (Decimal(high) - Decimal(1 / 2)) * Decimal(2)
                tag = (Decimal(tag) - Decimal(1 / 2)) * Decimal(2) + Decimal(2**(-tag_length)*string[next])
                next = next + 1

            if high < Decimal(1 / 2):
                low = Decimal(low) * Decimal(2)
                high = Decimal(high) * Decimal(2)
                tag = Decimal(tag) * Decimal(2) + Decimal(2 ** (-tag_length) * string[next])
                next = next + 1
        Range = Decimal(high) - Decimal(low)
    result = np.array(result)
    return result



filename = r"./data/val_set.txt"

try:
    with open(filename, 'rb+') as file:
        data = file.read()
        file.close()
except OSError:
    print("File not found")

# Converting into ascii
data = data.decode('utf-8')

data_split = data.split()

binary_data = [10 if i == '.' else int(i) for i in data_split]

vocabs = list(set(binary_data))
length_bytes ,vocab_size= len(binary_data),len(vocabs)
print ('data has %d characters, %d unique.' % (length_bytes, vocab_size))
print(vocabs)

b_input = np.array(binary_data)
print(len(b_input))
p = np.load("MERL_prob.npy" )
height,classes = p.shape
print(p.shape)

string = encoder(p[1:],b_input[1:])
#print(string)
print('encoded bits:',len(string))
print("Total size of compressed file (bytes): ", int(len(string)/8.0))
encoded_end = time.time()
print('encoding time:',encoded_end-start)

de = decoder(p[1:], string[1:])
decoded_end = time.time()
print('decoding time:',decoded_end-encoded_end)
correct_code = np.equal(b_input, de[:len(b_input)])
accuracy = np.sum(correct_code.astype(float))/height
print('decode accuracy:',accuracy*100.0)



