import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import requests
from pathlib import Path


if Path('helper_functions.py').is_file():
    print('already exists, skipping download')
else:
    print('downloading')
    request = requests.get('https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py')
    with open('helper_functions.py', 'wb') as f:
        f.write(request.content)

from helper_functions import accuracy_fn



word_pairs = [('cat', 'felino'), ('tree', 'arbol'), ('sky', 'cielo'), ('water', 'agua'), ('food', 'comida'), ('house', 'casa'), ('bird', 'pajaro'), ('earth', 'tierra'), ('fire', 'fuego'), ('sea', 'mar'), ('rain', 'lluvia'), ('road', 'camino'), ('fish', 'pez'), ('friend', 'amigo'), ('milk', 'leche'), ('car', 'coche'), ('brother', 'hermano'), ('sister', 'hermana'), ('bread', 'pan'), ('sand', 'arena'), ('chair', 'silla'), ('table', 'mesa'), ('window', 'ventana'), ('door', 'puerta'), ('knife', 'cuchillo'), ('apple', 'manzana'), ('cloud', 'nube'), ('morning', 'manana'), ('evening', 'tarde'), ('king', 'rey'), ('glass', 'vaso'), ('fork', 'tenedor'), ('shirt', 'camisa'), ('bird', 'ave'), ('goat', 'cabra'), ('blue', 'azul'), ('red', 'rojo'), ('yellow', 'amarillo'), ('black', 'negro'), ('white', 'blanco'), ('orange', 'naranja'), ('purple', 'morado'), ('banana', 'platano'), ('village', 'pueblo'), ('room', 'habitacion'), ('hill', 'colina'), ('field', 'campo'), ('farm', 'granja'), ('sheep', 'oveja'), ('cow', 'vaca'), ('boat', 'barco'), ('ship', 'nave'), ('stone', 'piedra'), ('sky', 'cielo'), ('chicken', 'pollo'), ('finger', 'dedo'), ('hand', 'mano'), ('leg', 'pierna'), ('head', 'cabeza'), ('door', 'puerta'), ('Dog', 'Perro'), ('Apple', 'Manzana'), ('Book', 'Libro'), ('Car', 'Coche'), ('House', 'Casa'), ('Chair', 'Silla'), ('Window', 'Ventana'), ('Food', 'Comida'), ('Water', 'Agua'), ('Sky', 'Cielo'), ('Sleep', 'Dormir'), ('Road', 'Camino'), ('Love', 'Amor'), ('acrobatic', 'acrobatico'), ('admire', 'admirar'), ('African', 'africano'), ('animal', 'animal'), ('appear', 'aparecer'), ('arithmetic', 'aritmetica'), ('attention', 'atencion'), ('bicycle', 'bicicleta'), ('biography', 'biografia'), ('blouse', 'blusa'), ('captain', 'capitan'), ('circle', 'circulo'), ('color', 'color'), ('December', 'diciembre'), ('delicate', 'delicado'), ('depend', 'depender'), ('deport', 'deportar'), ('destroy', 'destruir'), ('direction', 'direccion'), ('directions', 'direcciones'), ('directly', 'directamente'), ('director', 'director'), ('disgrace', 'desgracia'), ('double', 'doble'), ('dragon', 'dragon'), ('dinosaur', 'dinosaurio'), ('enormous', 'enorme'), ('energy', 'energia'), ('enter', 'entrar'), ('escape', 'escapar'), ('exclaim', 'exclamar'), ('diamond', 'diamante'), ('dictator', 'dictador'), ('different', 'diferente'), ('common', 'comun'), ('concert', 'concierto'), ('confusing', 'confuso'), ('continent', 'continente'), ('coyote', 'coyote'), ('curious', 'curioso'), ('favorite', 'favorito'), ('fruit', 'fruta'), ('gallon', 'galon'), ('gas', 'gas'), ('giraffe', 'jirafa'), ('glorious', 'glorioso'), ('group', 'grupo'), ('guide', 'guia'), ('honor', 'honor'), ('immediately', 'inmediatamente'), ('immigrants', 'inmigrantes'), ('incurable', 'incurable'), ('independence', 'independencia'), ('information', 'informacion'), ('insects', 'insectos'), ('inspection', 'inspeccion'), ('interrupt', 'interrumpir'), ('invent', 'inventar'), ('investigate', 'investigar'), ('lemon', 'limon'), ('magic', 'magia'), ('manner', 'manera'), ('monument', 'monumento'), ('natural', 'natural'), ('necessity', 'necesidad'), ('nectar', 'nectar'), ('object', 'objeto'), ('occasion', 'ocasion'), ('October', 'octubre'), ('palace', 'palacio'), ('part', 'parte'), ('patience', 'paciencia'), ('piano', 'piano'), ('pioneer', 'pionero'), ('prepare', 'preparar'), ('present', 'presentar'), ('radio', 'radio'), ('rich', 'rico'), ('secret', 'secreto'), ('series', 'serie'), ('study', 'estudiar'), ('traffic', 'trafico'), ('triple', 'triple'), ('trumpet', 'trompeta'), ('version', 'version'), ('volleyball', 'voleibol'), ('splendid', 'esplendid')]
labels = [0] * 73 + [1] * 87  # 1 for cognates, 0 for non-cognates
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



# loading data
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
        i, (hidden2, i) = self.rnn(embedded2)

        hidden_cat = torch.cat((hidden1[-1], hidden2[-1]), dim = -1)

        output = self.sigmoid(self.fc(hidden_cat))
        return output

vocab_size = len(ascii_rep)
embed_size = 16
hidden_size = 32

model = CognateClassifier(vocab_size, embed_size, hidden_size, max_len)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(30):
    for word1, word2, label in dataLoader:
        model.zero_grad()
        optimizer.zero_grad()

        output = model(word1, word2).squeeze(dim = 1)
        loss = loss_function(output, label)


        loss.backward()
        optimizer.step()

test_word_pairs = [('beef', 'buey'), ('house', 'casa'), ('castle', 'castillo')]
test_X_tensor = [(torch.tensor(word_to_idx(word1, max_len)), torch.tensor(word_to_idx(word2, max_len))) for word1, word2, in test_word_pairs]

model.eval()

with torch.inference_mode():

    for i in range(15):
        randomvalue = random.randint(0, 120)
        w1, w2 = word_pairs[randomvalue][0], word_pairs[randomvalue][1]
        prediction = model(torch.tensor(word_to_idx(w1, max_len)).unsqueeze(0), torch.tensor(word_to_idx(w1, max_len)).unsqueeze(0))
        print(f"Trained prediction for '{w1} {w2}': {prediction.item():.4f}")

    accuracy = 0
    for word1, word2, label in dataLoader:
        accuracy += accuracy_fn(torch.round(model(word1, word2).squeeze(dim=1)), label)

    for word1, word2 in test_X_tensor:
        prediction = model(word1.unsqueeze(0), word2.unsqueeze(0))
        print(f"Prediction for '{test_word_pairs.pop(0)}': {prediction.item():.4f}")
