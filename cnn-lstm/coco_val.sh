python src/build_vocab.py --caption_path data/annotations/captions_val2014.json --vocab_path ./data/vocab_val.pkl
python src/coco_val.py --sample_num 10  --model cnn-lstm