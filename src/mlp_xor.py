import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Dataset XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

configuraciones = [
    {"hidden_layer_sizes": (2,), "max_iter": 1000},
    {"hidden_layer_sizes": (4,), "max_iter": 1000},
    {"hidden_layer_sizes": (4, 4), "max_iter": 2000}
]

for i, config in enumerate(configuraciones, start=1):
    modelo = MLPClassifier(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        max_iter=config["max_iter"],
        random_state=1
    )

    modelo.fit(X, y)
    pred = modelo.predict(X)
    exactitud = accuracy_score(y, pred)

    print(f"\n--- Configuración {i} ---")
    print("Capas ocultas:", config["hidden_layer_sizes"])
    print("Iteraciones:", config["max_iter"])
    print("Predicciones:", pred)
    print("Exactitud:", exactitud)