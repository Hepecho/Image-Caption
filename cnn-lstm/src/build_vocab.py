import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.idx = 0
        

    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1
        return
        

    def __call__(self, word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return self.word2id['<unk>']
        

    def __len__(self):
        return self.idx
        

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    # print(coco.dataset.keys())  # dict_keys(['info', 'images', 'licenses', 'annotations'])
    # print(len(coco.anns.keys()))  # [..., 829717] 414113
    # print(coco.anns[828779])  # {'image_id': 58153, 'id': 828779, 'caption': 'This is a nice breakfast of eggs, a mini muffin, coffee, and orange juice.'}
    counter = Counter()

    ids = coco.anns.keys()
    # nltk.download('punkt_tab')

    for i in ids:
        caption = coco.anns[i]['caption']
        words = nltk.tokenize.word_tokenize(caption)
        counter.update(words)
    

    # If the word frequency is less than 'threshold', then the word is discarded.
    vocab_words = []
    for word, nums in counter.items():
        if nums >= threshold:
            vocab_words.append(word)

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for word in vocab_words:
        vocab.add_word(word)
    
    return vocab

def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)