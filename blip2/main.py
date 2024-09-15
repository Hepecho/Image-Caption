# pip install accelerate bitsandbytes
import torch
import requests
import argparse
import numpy as np
import os
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def blip2_api(image):
    # 如果无法连接huggingfac，命令行添加 export HF_ENDPOINT="https://hf-mirror.com" 
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")

    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    # # VQA
    # question = "how many dogs are in the picture?"
    # inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

    # out = model.generate(**inputs)
    # print(processor.decode(out[0], skip_special_tokens=True).strip())

    # Image-Caption
    
    image = image.convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return generated_text


def main(image_source):
    image_paths = []
    if os.path.isdir(image_source):
        names = os.listdir(image_source)
        image_paths = [os.path.join(image_source, name) for name in names]
    elif os.path.isfile(image_source):
        image_paths = [image_source]
    else:
        assert False, 'wrong args.image_source!'

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")

    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.convert('RGB')
        
         # 如果无法连接huggingfac，命令行添加 export HF_ENDPOINT="https://hf-mirror.com" 
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(f"{image_path} using blip2 8bit: {generated_text}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--image_source', type=str, default='../val_data/doc/', help='input image dir or path for generating caption')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    main(args.image_source)