import numpy as np
from src.pipeline import explain_anomaly

window = np.random.randn(30, 17).astype("float32") * 3
feature_names = [f"sensor_{i}" for i in range(1, 18)]
score, alert = explain_anomaly(window, feature_names, threshold=0.25)
print(f"score={score:.4f}, is_anomaly={alert is not None}")
print(alert)
