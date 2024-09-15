
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os
import argparse
import numpy as np


max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vit_api(image):
  # 如果无法连接huggingfac，命令行添加 export HF_ENDPOINT="https://hf-mirror.com" 
  model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
  feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
  tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

  model.to(device)

  if image.mode != "RGB":
    image.convert(mode="RGB")

  pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds[0]
  

def main(image_source):
  image_paths = []
  if os.path.isdir(image_source):
      names = os.listdir(image_source)
      image_paths = [os.path.join(image_source, name) for name in names]
  elif os.path.isfile(image_source):
      image_paths = [image_source]
  else:
      assert False, 'wrong args.image_source!'

  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]

  for image_path, caption in zip(image_paths, preds):
    print(f'{image_path} using vit-gpt2: {caption}]')



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--image_source', type=str, default='../val_data/doc/', help='input image dir or path for generating caption')
  args = parser.parse_args()

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  torch.backends.cudnn.deterministic = True

  # 如果无法连接huggingfac，命令行添加 export HF_ENDPOINT="https://hf-mirror.com" 
  model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
  feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
  tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

  model.to(device)

  main(args.image_source)

  
"""
image [R-C.jpg], caption [two people sitting on a park bench reading a book]
image [fox.jpg], caption [a brown and white dog laying in the grass]
image [p1.jpg], caption [a woman standing in a field with flowers]
image [e1.jpg], caption [a large body of water with a bridge over it]
image [n1.jpg], caption [a man and woman dressed in black and white are hugging each other]
image [c2.jpg], caption [a doll of a giraffe in a flower arrangement]
image [c1.jpg], caption [a woman standing next to a statue of an elephant]
image [n3.jpg], caption [a man and a woman standing on top of a surfboard]
image [p2.jpg], caption [a woman is sleeping in a car with a blanket over her head]
image [n2.jpg], caption [a man laying on the ground in front of a group of people]
image [c4.jpg], caption [two men standing next to each other in a field]
image [c3.jpg], caption [a woman holding an umbrella over her head]

Ability: 单一object和常见action关系能准确描述
Style: 量词 + 主体 + 动作/关系
Limitation: 多个object无法统一或分别描述；非摄影作品的理解有偏差和错漏
"""