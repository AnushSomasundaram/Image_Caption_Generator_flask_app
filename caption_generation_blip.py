# BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation / Model by Sales Force

# Model taken from https://huggingface.co/Salesforce/blip-image-captioning-base 


import requests
from PIL import Image

# get pretrained model from huggingface 
from transformers import BlipProcessor, BlipForConditionalGeneration


def generate_caption_BLIP_SALES_FORCE(image_location):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cpu")

    #img_url = '/content/sample_data/Screenshot 2023-04-11 at 2.02.08 PM.png' 
    #raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    image  = Image.open(image_location)
    raw_image = image.convert('RGB')
    # conditional image captioning
    text = "A photograph of"
    
    # the device that the process will be run on is specified here along with the model
    inputs = processor(raw_image, text, return_tensors="pt").to("cpu")

    out1 = model.generate(**inputs)
    
    #print(processor.decode(out1[0], skip_special_tokens=True))
    # >>> a photography of a woman and her dog

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cpu")

    out2 = model.generate(**inputs)
    
    #print(processor.decode(out2[0], skip_special_tokens=True))

    outputs = [processor.decode(out1[0], skip_special_tokens=True),processor.decode(out2[0], skip_special_tokens=True)]

    return outputs

#captions = generate_caption("/Users/software/Desktop/Listed/Screenshot 2023-04-11 at 4.40.46 PM.png")

#print(captions)