from ast import Tuple

import numpy as np  # type: ignore

from lizardnn.layers import Linear, ReLU, Sequential
from lizardnn.loss import BCELoss, CrossEntropyLoss
from lizardnn.optim import Adam
from lizardnn.tensor import Tensor


# ------------- tdataset --------------
def make_blobs(n_per_class: int = 100, std: float = 0.4, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    centers = np.array([[2, 0], [-2, 0], [0, 2]], dtype=np.float32)
    xs, ys = [], []
    for i, c in enumerate(centers):
        xs.append(c + rng.normal(scale=std, size=(n_per_class, 2)).astype(np.float32))
        ys.append(np.full((n_per_class,), i, dtype=np.int64))
    return np.vstack(xs), np.hstack(ys)


X_np, y_np = make_blobs()
X = Tensor(X_np, requires_grad=False)  # shape (N, 2)
y = Tensor(y_np, requires_grad=False)  # class indices 0-2

# ------------------- model ----------------------
model = Sequential(
    [
        Linear(2, 32),
        ReLU(),
        Linear(32, 16),
        ReLU(),
        Linear(16, 3),
    ]
)
criterion = BCELoss()
opt = Adam(model.get_parameters(), lr=0.01)

# ------------------- training loop ----------------------
for epoch in range(201):
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    opt.step()

    if epoch % 40 == 0:
        preds = np.argmax(logits.data, axis=1)
        acc = (preds == y_np).mean() * 100
        print(f"epoch {epoch:3d}  loss {loss.data:.4f}  acc {acc:5.1f}%")
# Example
test_pt = Tensor([[0.0, 2.5]], requires_grad=False)
test_logits = model(test_pt)
print("\nlogits for [0, 2.5]:", test_logits.data)
print("predicted class:", np.argmax(test_logits.data))
