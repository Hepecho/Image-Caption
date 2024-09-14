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

def sample_api(image, vocab, model_id):
    """
    api for coco_val
    image: raw PIL image
    vocab: train dataset vocab
    model_id: checkpoint model id, dtype=int
    """
    image = image.resize([224, 224])
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    image = transform(image).to(device)
    # Build models
    config = OmegaConf.load('config/default.yaml')
    encoder = EncoderCNN(config.embed_size).to(device)
    decoder = DecoderRNN(config.embed_size, config.hidden_size, len(vocab), config.num_layers).to(device)
    # Load the trained model parameters
    model_path = os.path.join('checkpoint/', f'model_{model_id}.pth')
    checkpoint = torch.load(model_path)
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.to(device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.to(device)
    
    # Generate an caption from the image
    features = encoder(image.unsqueeze(0))
    word_ids = decoder.sample(features, vocab)
    
    # Convert word_ids to words
    caption = []
    for word_id in word_ids:
        caption.append(vocab.id2word[word_id])
    
    # Print out the image and the generated caption
    caption_line = ' '.join(caption)
    # print(f"{model_path}: {caption_line}")
    return caption_line


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224])
    if image.mode != 'RGB':  # 灰度图转RGB
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

    # Prepare an image
    image = load_image(image_path, transform).to(device)

    for model_id in args.model_list:
        # Load the trained model parameters
        model_path = os.path.join('checkpoint/', f'model_{model_id}.pth')
        checkpoint = torch.load(model_path)
        encoder.load_state_dict(checkpoint['encoder'])  # 这里checkpoint可以看为字典，和之前保存的state相对应
        decoder.load_state_dict(checkpoint['decoder'])
        
        # Generate an caption from the image
        features = encoder(image.unsqueeze(0))
        word_ids = decoder.sample(features, vocab)
        
        # Convert word_ids to words
        caption = []
        for word_id in word_ids:
            caption.append(vocab.id2word[word_id])
        
        # Print out the image and the generated caption
        caption_line = ' '.join(caption)
        print(f"{model_path}: {caption_line}")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--image', type=str, default='../vit-gpt2/doc/n1.jpg', help='input image for generating caption')
    parser.add_argument('--model_list', nargs='+', default=list(range(5, 16)), help='trained model path list')
    parser.add_argument('--config', type=str, default='config/default.yaml' , help='path for config file')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    main(args.image, config)

    """
    p2:
    checkpoint/model_4.pth: A woman is sitting on a bench in a room .
    checkpoint/model_5.pth: A woman is taking a picture of herself in the mirror .
    checkpoint/model_6.pth: Two women are looking at a large mirror .
    checkpoint/model_7.pth: A man is sitting on a bench with a dog .
    checkpoint/model_8.pth: A woman sitting on a bed with a cat on her lap .
    checkpoint/model_9.pth: Two people sitting on a bench looking at a laptop .
    checkpoint/model_10.pth: Two women sitting on a couch with their luggage .
    checkpoint/model_11.pth: Two women are looking at a large mirror .
    checkpoint/model_12.pth: A man is eating food from a tray on a table .
    checkpoint/model_13.pth: A woman is taking a picture of a child .
    n1:
    checkpoint/model_5.pth: A man on a skateboard performing a trick .
    checkpoint/model_6.pth: A man is standing on a skateboard on a ramp .
    checkpoint/model_7.pth: A man on a skateboard is performing a trick .
    checkpoint/model_8.pth: A man on a skateboard on a city street .
    checkpoint/model_9.pth: A man on a skateboard riding down a ramp .
    checkpoint/model_10.pth: A man on a skateboard performing a trick .
    checkpoint/model_11.pth: A man is standing on a skateboard on a sidewalk .
    checkpoint/model_12.pth: A man on a skateboard is on a ramp .
    checkpoint/model_13.pth: A man is skateboarding on a street .
    checkpoint/model_14.pth: A man on a skateboard jumping over a ramp .
    checkpoint/model_15.pth: A man riding a skateboard down a street .
    """