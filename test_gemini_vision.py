import os
from dotenv import load_dotenv
load_dotenv(override=True)
import google.generativeai as genai
from PIL import Image

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

img = Image.new('RGB', (60, 30), color = 'red')

models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.5-flash",
    "gemini-pro-vision"
]

for m in models:
    print(f"Testing model {m}")
    try:
        model = genai.GenerativeModel(m)
        response = model.generate_content(["Describe this image", img])
        print("Success:", response.text)
        break
    except Exception as e:
        print(f"ERROR: {e}")
