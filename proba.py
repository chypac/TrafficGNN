import PySimpleGUI as sg
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv

# Определение модели графовой нейронной сети
class TrafficGNN(torch.nn.Module):
    def __init__(self):
        super(TrafficGNN, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Функция для генерации случайного графа дорожной сети
def generate_graph(num_intersections, connection_prob):
    G = nx.erdos_renyi_graph(num_intersections, connection_prob, seed=42)
    traffic_data = np.random.rand(num_intersections, 1)
    for i in range(num_intersections):
        G.nodes[i]['traffic'] = traffic_data[i]
    return G, traffic_data

# Функция для визуализации графа
def visualize_graph(G):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', edge_color='gray')
    plt.title("Сгенерированная дорожная сеть")
    plt.show()

# Функция для обучения модели
def train_model(G, traffic_data, epochs=100):
    data = from_networkx(G)
    data.x = torch.tensor(traffic_data, dtype=torch.float)
    model = TrafficGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    target = torch.rand(len(G), 1)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
    return model

# Функция для предсказания времени переключения светофоров
def predict_traffic_lights(model, G, traffic_data):
    data = from_networkx(G)
    data.x = torch.tensor(traffic_data, dtype=torch.float)
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index)
        return predictions.numpy()

# Основная функция для запуска GUI
def main():
    sg.theme('LightBlue')

    layout = [
        [sg.Text('Количество перекрестков:'), sg.InputText('10', key='num_intersections')],
        [sg.Text('Вероятность соединения:'), sg.InputText('0.3', key='connection_prob')],
        [sg.Button('Сгенерировать граф'), sg.Button('Обучить модель'), sg.Button('Предсказать')],
        [sg.Output(size=(60, 10))]
    ]

    window = sg.Window('Управление дорожным движением с помощью GNN', layout)

    G = None
    traffic_data = None
    model = None

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == 'Сгенерировать граф':
            try:
                num_intersections = int(values['num_intersections'])
                connection_prob = float(values['connection_prob'])
                G, traffic_data = generate_graph(num_intersections, connection_prob)
                visualize_graph(G)
                print('Граф сгенерирован.')
            except ValueError:
                print('Пожалуйста, введите корректные числовые значения.')
        elif event == 'Обучить модель':
            if G is None or traffic_data is None:
                print('Сначала сгенерируйте граф.')
            else:
                model = train_model(G, traffic_data)
                print('Модель обучена.')
        elif event == 'Предсказать':
            if model is None:
                print('Сначала обучите модель.')
            else:
                predictions = predict_traffic_lights(model, G, traffic_data)
                print('Предсказанные времена переключения светофоров (в условных единицах):')
                for i, pred in enumerate(predictions):
                    print(f'Перекресток {i}: {pred[0]:.2f}')

    window.close()

if __name__ == '__main__':
    main()
