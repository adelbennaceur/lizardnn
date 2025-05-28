import numpy as np  # type: ignore

from lizardnn.layers import Linear, Sequential, Sigmoid
from lizardnn.loss import BCELoss
from lizardnn.optim import SGD
from lizardnn.tensor import Tensor

# ------------- toy data: 2 Gaussian blobs (binary) --------------
rng = np.random.default_rng(0)
N = 200
x0 = rng.normal(loc=[-2, 0], scale=0.6, size=(N, 2)).astype(np.float32)
x1 = rng.normal(loc=[2, 0], scale=0.6, size=(N, 2)).astype(np.float32)
X_np = np.vstack([x0, x1])
y_np = np.concatenate(
    [np.zeros(N, dtype=np.float32), np.ones(N, dtype=np.float32)]
).reshape(-1, 1)

X = Tensor(X_np, requires_grad=False)
y = Tensor(y_np, requires_grad=False)

# ------------------- model, loss, optimiser ----------------------
model = Sequential([Linear(2, 1), Sigmoid()])  # logistic regression
criterion = BCELoss()
opt = SGD(model.get_parameters(), lr=0.05)

# ------------------------ training loop --------------------------
for epoch in range(300):
    preds = model(X)  # probabilities in (0,1)
    loss = criterion(preds, y)
    loss.backward()
    opt.step()

    if epoch % 60 == 0:
        pred_label = (preds.data > 0.5).astype(np.float32)
        acc = (pred_label == y_np).mean() * 100
        print(f"epoch {epoch:3d}  loss {loss.data:.4f}  acc {acc:5.1f}%")

# quick sanity check on one point
pt = Tensor([[0.0, 0.0]], requires_grad=False)
print("\nP(class=1) at [0,0]:", model(pt).data)
