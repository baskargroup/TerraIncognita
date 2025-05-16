import os
import base64
import time
import json
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from together import Together
import openai
from openai import OpenAI as XAI
from google.generativeai import GenerativeModel as GeminiModel
import anthropic
print('Successful import with 1 workers with error')

# ---------- PROMPTS ----------
SYSTEM_PROMPT = '''
You are an entomologist. Your job is to classify insects into the following hierarchy:

Order:
Family:
Genus:
Species:

You must return only these four fields. If you are confident about a level, fill it in. If not, write "Unknown". Do not provide explanations, reasoning, or any other text.

Only return the output in the exact format above. No markdown, no commentary, no additional lines.
'''

USER_PROMPT = '''
Based on the provided image, identify the insect to the most specific taxonomic level possible. 
Provide the full hierarchy: Order, Family, Genus, and Species.
'''

# ---------- IMAGE ENCODING ----------
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def encode_image_to_base64_with_resize(image_path, max_dim=None):
    img = Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()

# ---------- TOGETHER.AI INFERENCE ----------
def run_together_inference(image_paths, model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", api_key="YOUR_TOGETHER_API_KEY"):
    client = Together(api_key=api_key)
    results = []

    def classify(image_path):
        def try_send(image_b64):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]}
            ]
            return client.chat.completions.create(model=model, messages=messages, max_tokens=400)

        try:
            image_b64 = encode_image_to_base64(image_path)
            try:
                response = try_send(image_b64)
            except Exception as e:
                if "bytes" in str(e):
                    print(f"🔁 Resizing {image_path} due to size...")
                    image_b64 = encode_image_to_base64_with_resize(image_path, max_dim=224)
                    response = try_send(image_b64)
                else:
                    raise e
            return {"image_path": image_path, "response": response.choices[0].message.content}
        except Exception as e:
            return {"image_path": image_path, "response": None, "error": str(e)}

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(classify, path) for path in tqdm(image_paths, desc="Submitting tasks")]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{model} Inference"):
            results.append(future.result())

    return results

# ---------- GPT-4o INFERENCE ----------
def run_gpt_inference(image_paths, model="gpt-4o", api_key="YOUR_OPENAI_API_KEY"):
    openai.api_key = api_key
    results = []

    def classify(image_path):
        def try_send(image_b64):
            return openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]}
                ]
            )

        try:
            image_b64 = encode_image_to_base64(image_path)
            try:
                response = try_send(image_b64)
            except Exception as e:
                if "bytes" in str(e):
                    print(f"🔁 Resizing {image_path} due to size...")
                    image_b64 = encode_image_to_base64_with_resize(image_path, max_dim=1024)
                    response = try_send(image_b64)
                else:
                    raise e
            return {"image_path": image_path, "response": response.choices[0].message.content}
        except Exception as e:
            return {"image_path": image_path, "response": None, "error": str(e)}

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(classify, path) for path in tqdm(image_paths, desc="Submitting tasks")]
        for future in tqdm(as_completed(futures), total=len(futures), desc=model + " Inference"):
            results.append(future.result())

    return results

# ---------- GROK / xAI INFERENCE ----------
def run_grok_inference(image_paths, model="grok-2-vision-latest", api_key="YOUR_XAI_API_KEY"):
    client = XAI(api_key=api_key, base_url="https://api.x.ai/v1")
    results = []

    def classify(image_path):
        def try_send(image_b64):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}},
                    {"type": "text", "text": USER_PROMPT}
                ]}
            ]
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.01,
                max_tokens=300
            )

        try:
            image_b64 = encode_image_to_base64(image_path)
            try:
                response = try_send(image_b64)
            except Exception as e:
                if "bytes" in str(e):
                    print(f"🔁 Resizing {image_path} due to size...")
                    image_b64 = encode_image_to_base64_with_resize(image_path, max_dim=224)
                    response = try_send(image_b64)
                else:
                    raise e
            return {"image_path": image_path, "response": response.choices[0].message.content}
        except Exception as e:
            print(f"❌ Error with image {image_path}: {e}")
            return {"image_path": image_path, "response": None, "error": str(e)}

# ---------- GEMINI INFERENCE ----------
def run_gemini_inference(image_paths, model='gemini-1.5-flash', api_key="YOUR_GEMINI_API_KEY"):
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model)
    results = []

    def classify(image_path):
        try:
            try:
                img = Image.open(image_path).convert("RGB")
                response = model.generate_content(
                    contents=[SYSTEM_PROMPT + USER_PROMPT, img],
                    generation_config={"temperature": 0.1}
                )
                return {"image_path": image_path, "response": response.text}
            except Exception as e:
                print(f"🔁 Resizing {image_path} due to error: {e}")
                img = Image.open(image_path).convert("RGB")
                img.thumbnail((1024, 1024))
                response = model.generate_content(
                    contents=[SYSTEM_PROMPT + USER_PROMPT, img],
                    generation_config={"temperature": 0.1}
                )
                return {"image_path": image_path, "response": response.text}
        except Exception as e:
            print(e)
            return {"image_path": image_path, "response": None, "error": str(e)}

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(classify, path) for path in tqdm(image_paths, desc="Submitting tasks")]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Gemini Inference"):
            results.append(future.result())

    return results

# ---------- CLAUDE / ANTHROPIC INFERENCE ----------
def run_claude_inference(image_paths, model="claude-3-opus-20240229", api_key="YOUR_ANTHROPIC_API_KEY"):
    client = anthropic.Anthropic(api_key=api_key)
    results = []

    def classify(image_path):
        def try_send(image_b64):
            return client.messages.create(
                model=model,
                max_tokens=300,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": SYSTEM_PROMPT + USER_PROMPT},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}}
                    ]}
                ]
            )

        try:
            with open(image_path, "rb") as img_file:
                image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
            try:
                response = try_send(image_b64)
            except Exception as e:
                if "bytes" in str(e):
                    print(f"🔁 Resizing {image_path} due to size...")
                    image_b64 = encode_image_to_base64_with_resize(image_path, max_dim=1024)
                    response = try_send(image_b64)
                else:
                    raise e
            return {"image_path": image_path, "response": response.content[0].text}
        except Exception as e:
            print(e)
            return {"image_path": image_path, "response": None, "error": str(e)}

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(classify, path) for path in tqdm(image_paths, desc="Submitting tasks")]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Claude Inference"):
            results.append(future.result())

    return results

# ---------- UTILITY TO GATHER IMAGES ----------
def get_image_paths(root_folder):
    image_paths = []
    for folder in os.listdir(root_folder):
        if folder.startswith("."):
            continue
        for file in os.listdir(os.path.join(root_folder, folder)):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root_folder, folder, file))
    return image_paths
