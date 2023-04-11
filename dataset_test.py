from datasets import load_dataset
from torch.utils.data import DataLoader

trans_dataset = load_dataset("fashion_mnist")
loader = DataLoader(trans_dataset['train'])
print(trans_dataset['train'][0])
