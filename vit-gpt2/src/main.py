
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os
from os.path import join as ospj

# 如果无法连接huggingfac，命令行添加 export HF_ENDPOINT="https://hf-mirror.com" 
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
image_dir = '../doc'

def predict_step(image_dir):
  image_names = os.listdir(image_dir)
  images = []
  for name in image_names:
    i_image = Image.open(ospj(image_dir, name))
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return image_names, preds


image_names, tmp_preds = predict_step(image_dir) # ['two people sitting on a park bench reading a book', 'a brown and white dog laying in the grass']

for name, caption in zip(image_names, tmp_preds):
  print(f'image [{name}], caption [{caption}]')