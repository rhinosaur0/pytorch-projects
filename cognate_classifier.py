import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

word_pairs = [('Dog', 'Perro'), ('Apple', 'Manzana'), ('Book', 'Libro'), ('Car', 'Coche'), ('House', 'Casa'), ('Chair', 'Silla'), ('Window', 'Ventana'), ('Food', 'Comida'), ('Water', 'Agua'), ('Sky', 'Cielo'), ('Fire', 'Fuego'), ('Sleep', 'Dormir'), ('Road', 'Camino'), ('Night', 'Noche'), ('Love', 'Amor'), ('accident', 'accidente'), ('air', 'aire'), ('blouse', 'blusa'), ('captain', 'capitan'), ('disgrace', 'desgracia'), ('garden', 'jardin'), ('leader', 'lider'), ('observatory', 'observatorio'), ('totally', 'totalmente'), ('volleyball', 'voleibol'), ('splendid', 'esplendido'), ('lion', 'leon'), ('investigate', 'investigar'), ('giraffe', 'jirafa'), ('common', 'comun')]
labels = [0] * 15 + [1] * 15  # 1 for cognates, 0 for non-cognates
max_len = max(max(len(word1), len(word2)) for word1, word2 in word_pairs)

# convert to ascii
ascii_rep = {chr(i + 97): i + 1 for i in range(26)}
ascii_rep['empty'] = 0


def word_to_idx(word, max_len):
  indices = [ascii_rep[char] for char in word.lower()]
  return indices + [ascii_rep['empty']] * (max_len - len(indices))



class CognateDataset(Dataset):
  def __init__(self, word_pairs, labels, max_len):
    self.word_pairs = word_pairs
    self.labels = labels
    self.max_len = max_len

  def __len__(self):
    return len(self.word_pairs)

  # return (word1, word2, label) for dataLoader to retrieve in training
  def __getitem__(self, idx):
    word1, word2 = self.word_pairs[idx]
    word1_indices = word_to_idx(word1, self.max_len)
    word2_indices = word_to_idx(word2, self.max_len)
    label = self.labels[idx]
    return torch.tensor(word1_indices), torch.tensor(word2_indices), torch.tensor(label, dtype = torch.float32)


dataset = CognateDataset(word_pairs, labels, max_len)
dataLoader = DataLoader(dataset, batch_size = 2, shuffle = True)


class CognateClassifier(nn.Module):
  def __init__(self, vocab_size, embed_size, hidden_size, max_len):
    super(CognateClassifier, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx = ascii_rep['empty']) # optional padding_idx
    self.rnn = nn.LSTM(embed_size, hidden_size, batch_first = True)
    self.fc = nn.Linear(2 * hidden_size, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, word1, word2):
    #word1 has shape of (batch_size, max_length)
    embedded1 = self.embedding(word1) # shape of (batch_size, max_length, embed_size)
    i, (hidden1, i) = self.rnn(embedded1)

    embedded2 = self.embedding(word2)
    i, (hidden2, i) =self.rnn(embedded2)

    hidden_cat = torch.cat((hidden1[-1], hidden2[-1]), dim = -1)

    output = self.sigmoid(self.fc(hidden_cat))
    return output

vocab_size = len(ascii_rep)
embed_size = 16
hidden_size = 32
weight_decay_val = 0.00001

model_0 = CognateClassifier(vocab_size, embed_size, hidden_size, max_len)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = weight_decay_val)

for epoch in range(30):
  for word1, word2, label in dataLoader:
    model_0.zero_grad()
    optimizer.zero_grad()

    output = model_0(word1, word2).squeeze(dim = 1)
    loss = loss_function(output, label)


    loss.backward()
    optimizer.step()

test_word_pairs = [('beef', 'buey'), ('house', 'casa'), ('castle', 'castillo')]
test_X_tensor = [(torch.tensor(word_to_idx(word1, max_len)), torch.tensor(word_to_idx(word2, max_len))) for word1, word2, in test_word_pairs]

model.eval()

with torch.inference_mode():
  for w1, w2 in word_pairs:
    prediction = model_0(torch.tensor(word_to_idx(w1, max_len)).unsqueeze(0), torch.tensor(word_to_idx(w1, max_len)).unsqueeze(0))
    print(f"Trained prediction for '{word_pairs.pop(0)}': {prediction.item():.4f}")

with torch.inference_mode():
  for word1, word2 in test_X_tensor:
    prediction = model_0(word1.unsqueeze(0), word2.unsqueeze(0))
    print(f"Prediction for '{test_word_pairs.pop(0)}': {prediction.item():.4f}")
