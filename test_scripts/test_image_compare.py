import time

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
import base64
import os
from PIL import Image
from icecream import ic
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

def init_model(model_name = "google/gemma-3-4b", temperature=0.0):
    return init_chat_model(model=model_name,
    # model="ollama:qwen2.5vl:3b",
    # model="ollama:gemma3:latest",
    model_provider="openai",
    # base_url="http://snaisprod01:8023/v1",
    base_url="http://192.168.1.10:8000/v1",
    # base_url="http://192.168.1.10:8080/v1",
    # base_url="http://snaisprod01:8024/v1",
    # base_url="http://snaisprod01:11434",
    api_key="EMPTY",
    temperature=0.0)


def encode_image(pil_image):
    try:
        type_img = pil_image.format
        from io import BytesIO
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG" if type_img is None else type_img)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        raise Exception(f"Error encoding image: {str(e)}")



def get_mime_type(pil_image):
    """Determine MIME type based on PIL image format"""
    format_map = {
        'PNG': 'image/png',
        'JPEG': 'image/jpeg',
        'JPG': 'image/jpeg',
        'GIF': 'image/gif',
        'WEBP': 'image/webp'
    }
    return format_map.get(pil_image.format, 'image/png')


device = "cpu"
model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14")
model.eval().to(device)

transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

#
def embedding_img_dino(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img[:, :, ::-1])
    tensor = transform(img).unsqueeze(0).to(device)
    embedding = model(tensor)  # [1, 768]
    embedding = F.normalize(embedding, p=2, dim=1)
    # return embedding.detach().numpy()[0]
    return embedding
if __name__ == "__main__":


    # img_sop = cv2.imread("image_test/SOP/step1.JPG")
    #
    # img_sop_embeded = embedding_img_dino(img_sop)
    # print(img_sop_embeded.shape)
    #
    #
    # img_live = cv2.imread("image_test/REFERENCE_IMAGE/1.JPG")
    # img_live_embeded = embedding_img_dino(img_live)
    # print(img_live_embeded.shape)
    #
    # # sim_expected = float(np.dot(img_live_embeded, img_sop_embeded))
    # # print(sim_expected)
    # similarity = F.cosine_similarity(img_live_embeded, img_sop_embeded)
    #
    # print("Similarity:", similarity.item())
    # # img_1 = Image.open("image_test/step1.JPG")
    # # img_1_type = get_mime_type(img_1)
    # # print(img_1_type)
    # #
    # # # llm = init_model("")
    #
    # cv2.imshow("img_sop",cv2.resize(img_sop, (640, 480)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # llm = init_model("qwen3-vl-8b")

    prev_time = 0
    last_send = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resize_frame = cv2.resize(frame, (640, 480))

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(resize_frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # if curr_time - last_send > 2.0:
        #     base64_img = encode_image(Image.fromarray(frame))
        #     msg = HumanMessage(content=[
        #         {"type": "image_url",
        #          "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
        #         {"type": "text", "text": "What you see?"},
        #     ])
        #
        #     response = llm.invoke([msg]).content.strip()
        #     print(response)
        #     last_send = time.time()

        key = cv2.waitKey(1) & 0xFF
        cv2.imshow("Live", resize_frame)
        if key == ord('q'):
         break

    cap.release()
    cv2.destroyAllWindows()
