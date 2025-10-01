import torch
from pathlib import Path

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TRAIN_DATA_PATH = Path("")
TEST_DATA_PATH = Path("")

LEARNING_RATE = 1.5e-4
EPOCH = 100
BATCH_SIZE = 32


MODEL_WEIGHTS_PATH = Path('artifacts/vgg16_mnist.pth')
IMAGE_SIZE = (224, 224)
NORMALIZATION_MEAN = (0.1307, 0.1307, 0.1307)
NORMALIZATION_STD = (0.3081, 0.3081, 0.3081)
CLASS_NAMES = tuple(str(i) for i in range(10))

# git pull -p
# git merge danny
# git push origin danny -f

# git commit --amend

# git status
# git branch
# git reset --soft 9e34e7c
# git log --oneline
# git reflog
# git commit -m "garbage"