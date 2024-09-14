import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
import random
import argparse
import pandas as pd
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
from data_loader import CocoDataset
from sample import sample_api
import sys
sys.path.append("..")
from vit_gpt2 import vit_api
from blip2 import blip2_api


def coco_val(args):
    with open(args.vocab_val_path, 'rb') as f:
        vocab_val = pickle.load(f)

    with open(args.vocab_train_path, 'rb') as f:
        vocab_train = pickle.load(f)

    coco = CocoDataset(root=args.image_dir,
                       json=args.caption_path,
                       vocab=vocab_val)

    sample_ids = random.sample(list(range(len(coco))), args.sample_num)

    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)

    anns = {
        'pred': [],
        'real': [],
        'coco_id': sample_ids
    }

    for i in sample_ids:
        image, target = coco[i]
        image.save(os.path.join(output_dir, f'{i}.jpg'))
        if args.model == 'cnn-lstm':
            pred_caption = sample_api(image, vocab_train, model_id=19)
        elif args.model == 'vit-gpt2':
            pred_caption = vit_api(image)
        else:
            pred_caption = blip2_api(image)
        
        real_caption = []
        for word_id in target:
            real_caption.append(vocab_val.id2word[int(word_id.cpu().numpy())])
    
        real_caption = ' '.join(real_caption)
        anns['pred'].append(pred_caption)
        anns['real'].append(real_caption)
    dataframe = pd.DataFrame(anns)
    dataframe.to_csv(os.path.join(output_dir, 'anns.csv'),index=False, sep=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/annotations/captions_val2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--image_dir', type=str, default='data/val2014/',
                        help='directory for val images')
    parser.add_argument('--output_dir', type=str, default='../val_data/', help='directory for outputs')
    parser.add_argument('--model', type=str, default='cnn-lstm', help='image-caption api choice, select from [cnn-lstm, vit-gpt2, blip2]')
    parser.add_argument('--vocab_val_path', type=str, default='./data/vocab_val.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--vocab_train_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--sample_num', type=int, default=10, 
                        help='sample num from COCO val data')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    assert args.model in ['cnn-lstm', 'vit-gpt2', 'blip2'], 'model must in [cnn-lstm, vit-gpt2, blip2]'

    coco_val(args)