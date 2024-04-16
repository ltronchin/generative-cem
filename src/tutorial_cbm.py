# Credits
# - https://towardsdatascience.com/implement-interpretable-neural-models-in-pytorch-6a5932bdb078
# - https://github.com/pietrobarbiero/pytorch_explain#quick-tutorial-on-concept-embedding-models

import torch
from torch_explain import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Parameters.

# Dataset.
x, c, y = datasets.xor(500)
x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(x, c, y, test_size=0.33, random_state=42)

# Model.
concept_encoder = torch.nn.Sequential(
    torch.nn.Linear(x.shape[1], 10),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(10, 8),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(8, c.shape[1]),
    torch.nn.Sigmoid(),
)
task_predictor = torch.nn.Sequential(
    torch.nn.Linear(c.shape[1], 8),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(8, 1),
)
model = torch.nn.Sequential(concept_encoder, task_predictor)

# Training.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
loss_form_c = torch.nn.BCELoss()
loss_form_y = torch.nn.BCEWithLogitsLoss()
model.train()
for epoch in range(2001):
    optimizer.zero_grad()

    # generate concept and task predictions
    c_pred = concept_encoder(x_train)
    y_pred = task_predictor(c_pred)

    # update loss
    concept_loss = loss_form_c(c_pred, c_train)
    task_loss = loss_form_y(y_pred, y_train)
    loss = concept_loss + 0.2*task_loss

    loss.backward()
    optimizer.step()

# Evaluate.
c_pred = concept_encoder(x_test)
y_pred = task_predictor(c_pred)

concept_accuracy = accuracy_score(c_test, c_pred > 0.5)
task_accuracy = accuracy_score(y_test, y_pred > 0)

c_different = torch.FloatTensor([0, 1])
print(f"f({c_different}) = {int(task_predictor(c_different).item() > 0)}")

c_equal = torch.FloatTensor([1, 1])
print(f"f({c_different}) = {int(task_predictor(c_different).item() > 0)}")

print('May the force be with you!')