
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

import matplotlib.pyplot as plt
import model

model = model.LetNet5()


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# How the data is accessed.
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
# How the data is retrieved in batched.
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory= True)

test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


optimer = optim.AdamW(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 5

train_losses = []
train_accuracies = []

for epoch in tqdm(range(num_epochs)):

    running_loss = 0.0
    correct = 0
    total = 0
    model.train()

    for data, labels in train_dataloader:

        outputs = model(data)
        loss = loss_fn(outputs, labels)

        optimer.zero_grad()
        loss.backward()
        optimer.step()

        running_loss += loss.item()
        _, predite = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predite==labels).sum().item()

    avg_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct/total
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] "f"Loss: {avg_loss:.4f} "f"Accuracy: {accuracy:.2f}%")
    
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, labels in test_dataloader:
            outputs = model(data)
            _, predited = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predited == labels).sum().item()
    test_acc = 100* correct/total

    print(f"Test Accuracy: {test_acc:.2f}%")


plt.plot(train_losses, label='Train Loss')
plt.plot(train_accuracies, label='Train Acc')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()

# if __name__ == "__main__":
#     images, label = next(iter(train_dataloader))
#     print(images.shape)
#     print(label)

#     plt.imshow(images[0][0], cmap="gray")
#     plt.title(f"Label:{label[0].item()}")
#     plt.axis("off")
#     plt.show()