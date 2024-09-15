import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.vocab = vocab
        self.transform = transform
        self.ids = list(self.coco.anns.keys())


    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        img_id = self.coco.anns[self.ids[index]]['image_id']
        img_descriptor = self.coco.loadImgs(img_id)
        image = Image.open(os.path.join(self.root, img_descriptor[0]['file_name']))
        if image.mode != 'RGB':  # 转RGB
            image = image.convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        # print(image.shape) torch.Size([3, 224, 224])

        # Convert caption (string) to word ids.
        caption = self.coco.anns[self.ids[index]]['caption']
        words = nltk.tokenize.word_tokenize(caption)
        target = []
        for word in words:
            target.append(self.vocab(word))

        # print(target) [151, 135, 15, 234, 1456, 14, 3434, 90, 15, 1432, 7530, 90, 911, 90, 7, 581, 4441, 20]
        target = torch.Tensor(target)
        # print(target.shape)
        
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, crop_size, crop_size).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        captions: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: x[1].shape[0], reverse=True)  # 满足pack_padded_sequence传参要求
    images, captions = zip(*data)
    images = list(images)
    captions = list(captions)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    max_length = captions[0].shape[0]
    captions_tensor = torch.zeros([len(captions), max_length], dtype=torch.int)
    lengths = []

    for i, caption in enumerate(captions):
        captions_tensor[i, :caption.shape[0]] = caption
        lengths.append(caption.shape[0])

    return images, captions_tensor, lengths
    

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    # image, target = coco[0]
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    root = 'data/resized2014'
    json = 'data/annotations/captions_train2014.json'
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    transform = transforms.Compose([ 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    batch_size = 16
    shuffle = True
    num_workers = 2
    data_loader = get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers)
    # for batch in data_loader:
    #     images, targets, lengths = batch
    #     print(images.shape)
    #     print(targets[0])
    #     print(lengths[:10])
    # torch.Size([16, 3, 224, 224])
    # tensor([   4,  384,  135, 1494,  116,   34,  798,   23,   15,  369,   80,  166, 1673,   20], dtype=torch.int32)
    # [14, 13, 13, 12, 12, 11, 11, 11, 10, 10]
