from importlib.metadata import requires

import torch

if torch.cuda.is_available():
    print("[INFO] CUDA Selected")
    DEVICE = torch.device("cuda")

else:
    print("[INFO] CPU Selected")
    DEVICE = torch.device("cpu")

BATCH_SIZE = 64
INPUT_SIZE = 1000
HIDDEN_SIZE = 100
OUTPUT_SIZE = 10

# Input setting
# BATCH_SIZE : It's a data that for calculating and update parameter
x = torch.randn(BATCH_SIZE,
                INPUT_SIZE,
                device=DEVICE,
                dtype=torch.float,
                requires_grad=False)

# Output setting
y = torch.randn(BATCH_SIZE,
                OUTPUT_SIZE,
                device=DEVICE,
                dtype=torch.float,
                requires_grad=False)

# Update parameter settting
w1 = torch.randn(INPUT_SIZE,
                 HIDDEN_SIZE,
                 device=DEVICE,
                 dtype=torch.float,
                 requires_grad=True)

w2 = torch.randn(HIDDEN_SIZE,
                 OUTPUT_SIZE,
                 device=DEVICE,
                 dtype=torch.float,
                 requires_grad=True)

# THIS SCALAR MAKE GREAT AGAIN
# learning rate is caculating for any gradient to make sure their problem.
learning_rate = 1e-6

# run this loop for 500 times to updating parameter
for t in range(1, 501):
    # Predict result number, x & parameter w1 product output
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # compare predict output and label output. So I like a y_pred - y_pred^2
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 0:  # for monitoring
        print("Iteration: ", t, "\t", "Loss: ", loss.item())
    loss.backward()

    with torch.no_grad(
    ):  # okay. this part is little bit diffcult. Calculating gradient
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        # print("w1: ", w1, " w2: ", w2)

        w1.grad.zero_()
        w2.grad.zero_()
