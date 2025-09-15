import torch
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
from transformers import OFATokenizer, OFAModel

def ofa_model(path):

    # Load tokenizer and model from local clone
    MODEL_DIR = "./ofa-base"
    tokenizer = OFATokenizer.from_pretrained(MODEL_DIR)
    model = OFAModel.from_pretrained(MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # ✅ Use a reliable image URL
    url = path 
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Text prompt (captioning)
    prompt = " what does the image describe?"
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    # Run generation
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            patch_images=image_tensor,
            num_beams=5,
            max_length=16,
            no_repeat_ngram_size=3
        )

    # Decode result
    caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption
if __name__ == '__main__':
    print("Caption:", ofa_model("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"))