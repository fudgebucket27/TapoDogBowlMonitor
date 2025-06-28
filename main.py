import cv2
import base64
import os
import requests
from dotenv import load_dotenv
load_dotenv()

# Configuration

TAPO_IP   = os.getenv("TAPO_IP") 
TAPO_USER = os.getenv("TAPO_USER") 
TAPO_PASS = os.getenv("TAPO_PASS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

IMAGE_PATH = "images/dog_water_bowl.jpg"
CROP_PATH = "images/dog_water_bowl_cropped.jpg"
RTSP_URL   = f"rtsp://{TAPO_USER}:{TAPO_PASS}@{TAPO_IP}:554/stream1" 

def capture_snapshot():
    print("Capturing snapshot of the dog water bowl...")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG) #OpenCV is overkill to just get a single frame but I might use it for something else later on xD
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("Snapshot failed. Check your Tapo camera account credentials.")


    cv2.imwrite(IMAGE_PATH, frame)
    img = cv2.imread(IMAGE_PATH)
    h, w = img.shape[:2]
    x0, x1 = int(w*0.33), int(w*0.42)
    y0, y1 = int(h*0.60), int(h*0.72)
    crop   = img[y0:y1, x0:x1]
    cv2.imwrite(CROP_PATH, crop)

    print("Captured snapshot successfully")

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyse_image(base64_image):
    print("Sending image to OpenAI or analysis...")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = (
        """
        You are an expert system designed to monitor a pet's water bowl
        I will provide you with an image of a dog water bowl. 
        You are to only look at the water bowl to the right.
        Water will appear in the centre of the water bowl as a different colour or shade compared to the bowl's inner plastic wall.
        Ignore reflections.
        Your task is to determine the water level in the bowl in terms of percentage and your confidence in that percentage.
        You must run your analysis 3 times and return the average percentage and confidence level. Only give me your full final anaylsis.
        Give reasons for your analysis and confidence level and any issues that may have affected your analysis.
        If you cannot determine the water level, please state that clearly.
        """
    )   

    payload = {
        "model": "gpt-4.1",
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt
                    },
                    { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_image}", "detail": "high" } }
                ]
            }
        ],
    }

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    result = resp.json()

    if "error" in result:
        raise RuntimeError(
            f"OpenAI error [{result['error'].get('code')}]: "
            f"{result['error']['message']}"
        )

    print("OpenAI analysis:\n", result["choices"][0]["message"]["content"])

if __name__ == "__main__":
    capture_snapshot()
    analyse_image(image_to_base64(CROP_PATH))
