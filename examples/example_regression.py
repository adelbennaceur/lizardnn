from typing import cast

import numpy as np  # type: ignore

from lizardnn.layers import Linear, Sequential
from lizardnn.loss import MSELoss
from lizardnn.optim import SGD
from lizardnn.tensor import Tensor

# fake data: y = 3x + 2  (with noise)
rng = np.random.default_rng(0)
x_np = rng.uniform(-1, 1, (128, 1)).astype(np.float32)
y_np = 3 * x_np + 2 + rng.normal(0, 0.05, (128, 1)).astype(np.float32)

x = Tensor(x_np, requires_grad=False)
y = Tensor(y_np, requires_grad=False)

model = Sequential([Linear(1, 1)])
criterion = MSELoss()
opt = SGD(model.get_parameters(), lr=0.1)

for epoch in range(200):
    preds = model(x)
    loss = criterion(preds, y)
    loss.backward()
    opt.step()

    if epoch % 40 == 0:
        print(f"epoch {epoch:3d}  loss {loss.data:.4f}")

print("learned weight:", model[0].weight.data.ravel())  # type: ignore
print("learned bias :", model[0].bias.data)  # type: ignore
