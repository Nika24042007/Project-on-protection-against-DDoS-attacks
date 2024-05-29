import pandas as pd
import numpy as np
import ipaddress
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Пример данных
data = {
    'sourceIP': ['192.168.0.1', '2001:0db8:85a3:0000:0000:8a2e:0370:7334'],
    'destinationIP': ['10.0.0.1', '2001:0db8:85a3:0000:0000:8a2e:0370:7335'],
    'packetCount': [100, 200],
    'byteCount': [1000, 2000],
    'duration': [10, 20]
}

df = pd.DataFrame(data)

def ip_to_one_hot(ip):
    try:
        ip_obj = ipaddress.ip_address(ip)
        packed_ip = ip_obj.packed
        one_hot_encoded = []
        for byte in packed_ip:
            one_hot = [0] * 256
            one_hot[byte] = 1
            one_hot_encoded.extend(one_hot)
        return one_hot_encoded
    except ValueError:
        return [0] * (256 * 16)

# Кодируем IP-адреса
df['sourceIP_encoded'] = df['sourceIP'].apply(ip_to_one_hot)
df['destinationIP_encoded'] = df['destinationIP'].apply(ip_to_one_hot)

# Объединяем все признаки в один вектор
df['features'] = df.apply(lambda row: row['sourceIP_encoded'] + row['destinationIP_encoded'] + [row['packetCount'], row['byteCount'], row['duration']], axis=1)

# Разделяем данные на признаки и целевые значения (предположим, что byteCount - целевое значение)
X = np.array(df['features'].tolist())
y = np.array(df['byteCount'].tolist())

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразуем данные в тензоры
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Модель Encoder-Decoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        hidden, cell = self.encoder(src)
        outputs, hidden, cell = self.decoder(trg, hidden, cell)
        return outputs

# Параметры модели
input_dim = X_train.shape[1]
hidden_dim = 256
output_dim = 1

# Создание экземпляров Encoder и Decoder
encoder = Encoder(input_dim, hidden_dim)
decoder = Decoder(hidden_dim, output_dim)

# Создание модели Seq2Seq
model = Seq2Seq(encoder, decoder)

# Оптимизатор и функция потерь
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Обучение модели
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train.unsqueeze(0), y_train.unsqueeze(0))
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Тестирование модели
model.eval()
with torch.no_grad():
    test_output = model(X_test.unsqueeze(0), y_test.unsqueeze(0))
    test_loss = criterion(test_output, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')
