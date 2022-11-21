from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter

X,y = make_moons(n_samples=20000, noise=0.15)
X = X - X.mean(axis=0)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

X_test, X_validation, y_test, y_validation = train_test_split(
    X_test, y_test, test_size=0.3
)

class MoonsDataset(Dataset):
  def __init__(self):
    pass

  def __len__ (self):
    return len(y_train)

  def __getitem__(self, idx):
    item = X_train[idx]
    label = y_train[idx]
    return item, label


class skip_connection(torch.nn.Module):
  def __init__(self, in_dim, out_dim) -> None:
    super().__init__()
    self.layer = torch.nn.utils.spectral_norm(torch.nn.Linear(in_dim, out_dim))

  def forward(self, x):
    return x + self.layer(x)

class high_dim_proj(torch.nn.Module):
  def __init__(self, in_dim, out_dim) -> None:
    super().__init__()
    self.layer = torch.nn.Sequential(
       torch.nn.Linear(in_dim, 64),
       torch.nn.ReLU(),
       torch.nn.Linear(64, out_dim))

  def forward(self, x):
    return self.layer(x)

class MLP(torch.nn.Module):
  def __init__(self, dim_size):
    super().__init__()
    args = []
    for i, size in enumerate(dim_size):
      if i+1 == len(dim_size):
        args.append(torch.nn.Linear(size, 2))
      else:
        # args.append(torch.nn.Linear(size, dim_size[i+1]))
        args.append(skip_connection(size, dim_size[i+1]))
        args.append(torch.nn.ReLU())

    self.layers = torch.nn.Sequential(*args)

  def forward(self, features):
    return torch.sigmoid(self.layers(features))
  
  def extract_features(self, x):
    dic = {}
    for key, l in self.layers.named_children():
      x = l(x)
      dic[key] = x
    
    return dic

def train(model, epochs=50):

  optimiser = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

  for epoch in range(epochs):
    for batch in train_loader:
      features, labels = batch
      prediction = model(features.float())
      loss = F.cross_entropy(torch.squeeze(prediction).float(), labels)
      loss.backward()
      optimiser.step()
      optimiser.zero_grad()

def test(predictions, labels):
  correct = 0
  for i, el in enumerate(predictions):
    if el > 0.5 and labels[i] == 1:
      correct +=1
    elif el < 0.5 and labels[i] ==0:
      correct +=1
    else:
      pass

  accuracy = correct/len(predictions)

  print(f'Accuracy = {accuracy}%')


dataset = MoonsDataset()
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

n_layers = 3
model = MLP([2] * n_layers)
train(model)
data = model.extract_features(torch.from_numpy(X_validation).float())

def colourify(i):
  if i == 0:
    return 'red'
  else:
    return 'blue'

fig, ax = plt.subplots(nrows=n_layers - 1, ncols=2)

def extract_data(data):
  lists_linear = list(data)
  x = [x[0] for x in lists_linear]
  y = [x[1] for x in lists_linear]
  colors = [colourify(x) for x in y_validation]

  return (x, y), colors

for i in range(0, len(data)-2, 2):
    linear, color = extract_data(data[f'{i}'].detach().numpy())
    activ, color = extract_data(data[f'{i+1}'].detach().numpy())

    print(len(linear[0]))
    ax[int(i/2), 0].scatter(linear[0], linear[1], c=color)
    ax[int(i/2), 1].scatter(activ[0], activ[1], c=color)

plt.show()
    

# train(model)

val_prediction = model(torch.from_numpy(X_validation).float())
test(torch.squeeze(val_prediction), torch.from_numpy(y_validation))