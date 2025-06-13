import sys
from models.train_model import train

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py [train]")
        sys.exit(1)

    if sys.argv[1] == 'train':
        train()
