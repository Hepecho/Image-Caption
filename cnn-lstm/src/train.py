import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import time
from runx.logx import logx
from omegaconf import OmegaConf
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main(config):
    # Create model directory
    os.makedirs(config.model_dir, exist_ok=True)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(config.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(config.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(config.image_dir, config.caption_path, vocab, transform, config.batch_size, True, config.num_workers)

    # Build the models
    encoder = EncoderCNN(config.embed_size).to(device)
    decoder = DecoderRNN(config.embed_size, config.hidden_size, len(vocab), config.num_layers).to(device)
    if config.load_model != '':
        checkpoint = torch.load(os.path.join(config.model_dir, config.load_model))
        encoder.load_state_dict(checkpoint['encoder'])
        encoder.to(device)
        decoder.load_state_dict(checkpoint['decoder'])
        decoder.to(device)
        name = config.load_model.split('_')  # model_{epoch}_{i}.pth
        start_epoch = int(name[-2]) + 1
    else:
        start_epoch = 0

    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
                {'params': encoder.fc.parameters()},
                {'params': encoder.bn.parameters()},
                {'params': decoder.parameters()}
            ], lr=config.learning_rate)
    
    # Train the models
    encoder.train()
    decoder.train()
    
    for epoch in range(start_epoch, config.num_epochs):
        total_loss = 0
        start_time = time.time()
        for i, batch in enumerate(data_loader):
            # Set mini-batch dataset
            images, captions, lengths = batch
            images = images.to(device)
            captions = captions.to(device)
            
            # lengths = torch.Tensor(lengths, dtype=torch.int64)
            target = pack_padded_sequence(captions, lengths, batch_first=True)
            target = target.data.to(device)  # (real_L * B)
            # print(target[:20])
            # print(lengths[0])
            # print(vocab.id2word[4])
            
            # Forward, backward and optimize
            fearures = encoder(images)
            outputs = decoder(fearures, captions, lengths)  # (real_L * B, V)
            _, pred = outputs.max(1)
            
            loss = criterion(outputs, target.long())
            total_loss += loss.item() * len(lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % config.save_step == 0:
                state = {
                    'encoder':encoder.state_dict(),
                    'decoder':decoder.state_dict()
                }
                ckpt_path = os.path.join(config.model_dir, f'model_{epoch}_{i}.pth')
                torch.save(state, ckpt_path)
            if i % config.log_step == 0:
                logx.msg(f'Epoch: {epoch} | Batch {i} Loss: {loss.item():.4f}')

        # Print log info
        total_loss /= len(data_loader.dataset)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        state = {
            'encoder':encoder.state_dict(),
            'decoder':decoder.state_dict()
        }
        ckpt_path = os.path.join(config.model_dir, f'model_{epoch}_latest.pth')
        torch.save(state, ckpt_path)
        
        logx.msg(f'Epoch: {epoch} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logx.msg(f'Train Loss: {total_loss:.4f}')
            
        # Save the model checkpoints
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--config', type=str, default='config/default.yaml' , help='path for config file')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    logx.initialize(logdir=config.log_dir, coolname=False, tensorboard=False)
    logx.msg(str(args))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    main(config)