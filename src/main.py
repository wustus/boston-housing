
from dataset_loader import BostonHousingDataset
from torch.utils.data import DataLoader
from network import FFNetwork

import torch
import random

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

filter = []

train_ds = BostonHousingDataset("data/train.csv", filter=filter)
test_ds = BostonHousingDataset("data/test.csv", filter=filter, test=True)

batch_size = 8

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

net = FFNetwork(13 - len(filter), [25, 5], 1, dropout=0.1)
opt = torch.optim.SGD(net.parameters(), lr=5e-4)

epochs = 10_000

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

    if e % 100 == 0:
        print(f"Epoch {e}, Total Loss: {t_loss / len(train_ds):.4f}.")

    t_diff = 0
    net.eval()
    for x, y in test_dl:
        out = net(x).squeeze(-1)
        diff = abs(out - y).sum().item()
        t_diff += diff
    if e % 100 == 0:
        print(f"Epoch {e}, Average Difference: {t_diff / (len(test_ds)) :.4f}")
