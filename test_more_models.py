import os
from dotenv import load_dotenv
load_dotenv(override=True)
import google.generativeai as genai
from PIL import Image

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

img = Image.new('RGB', (60, 30), color = 'red')

models = [
    "models/gemini-2.0-flash-lite",
    "models/gemini-3.1-pro-preview",
    "models/gemma-3-27b-it",
    "models/gemini-flash-lite-latest"
]

for m in models:
    print(f"Testing model {m}")
    try:
        model = genai.GenerativeModel(m)
        response = model.generate_content(["Describe this image", img])
        print("Success:", response.text)
    except Exception as e:
        print(f"ERROR: {e}")
