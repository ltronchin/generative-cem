from torch_explain.datasets import trigonometry
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch_explain
import torch

# Parameters.
embedding_size = 8

# Dataset.
x, c, y = trigonometry(500)
x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(x, c, y, test_size=0.33, random_state=42)

# Model.
concept_encoder = torch.nn.Sequential(
    torch.nn.Linear(x.shape[1], 10),
    torch.nn.LeakyReLU(),
    torch_explain.nn.ConceptEmbedding(10, c.shape[1], embedding_size),
)
task_predictor = torch.nn.Sequential(
    torch.nn.Linear(c.shape[1]*embedding_size, 8),
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
    c_emb, c_pred = concept_encoder(x_train)
    y_pred = task_predictor(c_emb.reshape(len(c_emb), -1))

    # compute loss
    concept_loss = loss_form_c(c_pred, c_train)
    task_loss = loss_form_y(y_pred, y_train)
    loss = concept_loss + 0.2 * task_loss

    loss.backward()
    optimizer.step()

# Evaluate.
c_emb, c_pred = concept_encoder.forward(x_test)
y_pred = task_predictor(c_emb.reshape(len(c_emb), -1))

concept_accuracy = accuracy_score(c_test, c_pred > 0.5)
task_accuracy = accuracy_score(y_test, y_pred > 0)

print(f"Concept accuracy: {concept_accuracy:.2f}")
print(f"Task accuracy: {task_accuracy:.2f}")

print("May the force be with you!")

