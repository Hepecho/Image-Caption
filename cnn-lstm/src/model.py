import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        # resnet = models.resnet152(pretrained=True)
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    
    def forward(self, images):
        """Extract feature vectors from input images."""
        batch_size = images.shape[0]
        with torch.no_grad():
            features = self.resnet(images) # (B, 2048, 1, 1)
        
        features = features.reshape(batch_size, -1) # (B, 2048)
        
        features = self.fc(features)  # (B, embed_size)

        if batch_size > 1:
            features = self.bn(features)

        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.num_directions = 1 # 单向LSTM
        self.hidden_size = hidden_size  # H
        self.num_layers = num_layers  # L
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        # features: torch tensor of shape# (B, embed_size).
        # captions: torch tensor of shape (B, padded_length).
        # lengths: list; valid length for each padded caption
        embs = self.embedding(captions)  # (B, L, embed_size)
        
        inputs = torch.cat([features.unsqueeze(1), embs], dim=1)  # (B, 1 + L, embed_size)  把iamge附加到每个caption开头
        
        packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True)  # (real_L * B, embed_size)
        
        outputs, (_, _) = self.lstm(packed_inputs) # outputs.data (real_L * B, hidden_size)
        
        outputs = self.linear(outputs.data)  # (real_L * B, V)
        return outputs
    
    def sample(self, features, vocab, states=None):
        """Generate captions for given image features using greedy search."""
        output = features # (1, embed_size)
        
        max_seq_length = 20
        sampled_ids = []
        end_ids = [vocab('.'), vocab('<end>')]
        
        for i in range(max_seq_length):
            with torch.no_grad():
                if states is None:
                    output, states = self.lstm(output)
                else:
                    h_i, c_i = states
                    output, states = self.lstm(output, (h_i, c_i)) # output (1, hidden_size)
                
                pred = self.linear(output)
                _, pred_id = pred.max(1)  # word_id
                output = self.embedding(pred_id)  # output (1, embed_size)
            
            sampled_ids.append(int(pred_id[0].cpu().numpy()))

            if pred_id in end_ids:
                break

        return sampled_ids  # (L)

if __name__ == '__main__':
    import pickle
    from build_vocab import Vocabulary
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    encoder = EncoderCNN(256).to('cuda')
    decoder = DecoderRNN(256, 512, vocab_size, 1).to('cuda')

    # data
    from data_loader import get_loader
    root = 'data/resized2014'
    json = 'data/annotations/captions_train2014.json'
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    import torchvision.transforms as transforms
    transform = transforms.Compose([ 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    batch_size = 4
    shuffle = True
    num_workers = 1
    data_loader = get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers)
    for batch in data_loader:
        images, captions, lengths = batch
        images = images.to('cuda')
        captions = captions.to('cuda')
        # lengths = torch.Tensor(lengths, dtype=torch.int64)
        target = pack_padded_sequence(captions, lengths, batch_first=True).to('cuda')  # (real_L * B)
        print(f'target shape: {target.data.shape}')
        fearures = encoder(images)
        outputs = decoder(fearures, captions, lengths)
        exit()