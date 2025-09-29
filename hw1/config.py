import torch
from pathlib import Path

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TRAIN_DATA_PATH = Path("")
TEST_DATA_PATH = Path("")

LEARNING_RATE = 1.5e-4
EPOCH = 100
BATCH_SIZE = 32


