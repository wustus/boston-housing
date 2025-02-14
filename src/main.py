
from dataset_loader import BostonHousingDataset
from ds_transformer import DatasetTransformer
from torch.utils.data import DataLoader
from network import FFNetwork
from math import inf

import torch
import random
import os

# default: 42 else get from env
SEED = os.getenv("SEED")
SEED = 42 if SEED is None else int(SEED)
random.seed(SEED)

# default: true, else mostly truish
DEBUG = os.getenv("DEBUG")
DEBUG = False if "0" in str(DEBUG) else True

with open("data/boston.csv") as f:
    lines = f.readlines()[22:]
    lines = [(lines[i].strip() + " " + lines[i+1].strip()).replace("  ", " ") for i in range(0, len(lines)-1, 2)]
    random.shuffle(lines)

d_len = len(lines)
cols = "CRIM ZN INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO B LSTAT MEDV "

with open("data/train.csv", "w") as f:
    f.write(cols + "\n")
    f.write("\n".join(lines[:d_len//2]))

with open("data/test.csv", "w") as f:
    f.write(cols + "\n")
    f.write("\n".join(lines[d_len//2:]))

# filter out cols (not used)
filter = []

transformer = DatasetTransformer()

train_ds = BostonHousingDataset("data/train.csv", transformer, filter=filter)
test_ds = BostonHousingDataset("data/test.csv", transformer, filter=filter, test=True)

batch_size = 8

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

net = FFNetwork(len(train_ds.features()), [25, 5], 1, dropout=0.1)
opt = torch.optim.SGD(net.parameters(), lr=1e-4)

# way too many, stopping before that is a good idea
epochs = 5_000

best_avg_diff = inf

for e in range(1, epochs+1):

    net.train()
    t_loss = 0
    for x, y in train_dl:
        opt.zero_grad()
        out = net(x)
        loss = torch.nn.functional.mse_loss(out.squeeze(-1), y)
        t_loss += loss.item()
        loss.backward()
        opt.step()

    if e % 100 == 0 and DEBUG:
        print(f"Epoch {e}, Total Loss: {t_loss / len(train_ds):.4f}.")

    t_diff = 0
    mape = 0
    net.eval()
    for x, y in test_dl:
        out = net(x).squeeze(-1)
        diff = abs(out - y).sum().item()
        mape += abs((out - y) / y).sum()
        t_diff += diff
    if e % 100 == 0:
        avg_diff = t_diff / len(test_ds)
        if DEBUG:
            print(f"Epoch {e}, Average Difference: {avg_diff:.4f}, MAPE: {(1/len(test_ds)) * mape * 100.0:.2f}%")

        if avg_diff < best_avg_diff:
            best_avg_diff = avg_diff

print(f"Best Avg Diff: {best_avg_diff:.4f}")
