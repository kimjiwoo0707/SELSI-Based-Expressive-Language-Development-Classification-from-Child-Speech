import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class SpeechImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.subclass_info = []
        self.transform = transform

        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for super_idx, superclass in enumerate(self.classes):
            super_path = os.path.join(root_dir, superclass)

            subfolders = [f for f in os.listdir(super_path) if os.path.isdir(os.path.join(super_path, f))]

            if subfolders:
                for sub in subfolders:
                    sub_path = os.path.join(super_path, sub)
                    for file in os.listdir(sub_path):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.data.append(os.path.join(sub_path, file))
                            label = [1 if i == super_idx else 0 for i in range(len(self.classes))]
                            self.labels.append(label)
                            self.subclass_info.append(f"{superclass}/{sub}/{file}")

            else:
                for file in os.listdir(super_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.data.append(os.path.join(super_path, file))
                        label = [1 if i == super_idx else 0 for i in range(len(self.classes))]
                        self.labels.append(label)
                        self.subclass_info.append(f"{superclass}/{file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # one-hot
        subclass = self.subclass_info[idx]
        return image, label, subclass
