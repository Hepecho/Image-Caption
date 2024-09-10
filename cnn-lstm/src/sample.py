import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from omegaconf import OmegaConf
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224])
    if image.mode == 'L':  # 灰度图转RGB
        image = image.convert('RGB')
    
    if transform is not None:
        image = transform(image)

    return image

def main(image_path, config):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(config.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(config.embed_size).to(device)
    decoder = DecoderRNN(config.embed_size, config.hidden_size, len(vocab), config.num_layers).to(device)

    # Load the trained model parameters
    checkpoint = torch.load(args.model_path)
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.to(device) # 这里checkpoint可以看为字典，和之前保存的state相对应
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.to(device)

    # Prepare an image
    image = load_image(image_path, transform).to(device)
    
    # Generate an caption from the image
    features = encoder(image.unsqueeze(0))
    word_ids = decoder.sample(features, vocab)
    
    # Convert word_ids to words
    caption = []
    for word_id in word_ids:
        caption.append(vocab.id2word[word_id])
    
    # Print out the image and the generated caption
    print(' '.join(caption))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--image', type=str, default='../vit-gpt2/doc/p2.jpg', help='input image for generating caption')
    parser.add_argument('--model_path', type=str, default='checkpoint/model_3_latest.pth', help='trained model path')
    parser.add_argument('--config', type=str, default='config/default.yaml' , help='path for config file')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    main(args.image, config)