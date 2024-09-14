# pip install accelerate bitsandbytes
import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return generated_text