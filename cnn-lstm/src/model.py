import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        pass
    
    def forward(self, images):
        """Extract feature vectors from input images."""
        pass

class DecoderRNN(nn.Moddule):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        pass

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        pass
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        pass
        return sampled_ids