import streamlit as st
import requests  
import json
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import io
import torch

st.title('Practica 2')

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')


col1, col2 = st.columns(2)


url =  "https://stablediffusionapi.com/api/v4/dreambooth"

with col1:
    # Crear un campo de entrada de texto
    input_text = st.text_input('Ingrese algún texto')

    

    if st.button('Mostrar Imagen'):
        dict_input = {  
        "key":  "OJbkmhBeCe18RVSct56CL5XG6J994iNjF7k7CHcEgM7GZDGtEAsbdR2sZs4e",  
        "model_id":  "juggernaut-xl-v5",  
        "prompt":  "ultra realistic close up portrait ((beautiful pale cyberpunk female with heavy black eyeliner)), blue eyes, shaved side haircut, hyper detail, cinematic lighting, magic neon, dark red city, Canon EOS R3, nikon, f/1.4, ISO 200, 1/160s, 8K, RAW, unedited, symmetrical balance, in-frame, 8K",  
        "negative_prompt":  "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",  
        "width":  "512",  
        "height":  "512",  
        "samples":  "1",  
        "num_inference_steps":  "30",  
        "safety_checker":  "no",  
        "enhance_prompt":  "yes",  
        "seed":  None,  
        "guidance_scale":  7.5,  
        "multi_lingual":  "no",  
        "panorama":  "no",  
        "self_attention":  "no",  
        "upscale":  "no",  
        "embeddings":  "embeddings_model_id",  
        "lora":  "lora_model_id",  
        "webhook":  None,  
        "track_id":  None  
        }

        dict_input['prompt'] = input_text

        payload = json.dumps(dict_input)  
        
        headers =  {  
        'Content-Type':  'application/json'  
        }  
        
        response = requests.request("POST", url, headers=headers, data=payload)  
        
        im = 'https://pub-3626123a908346a7a8be8d9295f44e26.r2.dev/generations/0-b484c197-0523-4ea2-bcf4-5462c49b5ea8.png'
        # Si se hace clic en el botón, mostrar la imagen
        st.image(im,
                caption='Imagen Mostrada')
    #col1.write(response.json()['status'])
        
with col2:
   st.header("Image Classification")
   uploaded_file = st.file_uploader("Choose a file")
   if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(bytes_data))

        inputs = feature_extractor(images=image, return_tensors="pt")

        # Realizar la predicción
        outputs = model(**inputs)
        logits = outputs.logits

        # Obtener la clase con la mayor probabilidad
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

        # Retornar la clase predicha
        x = {"message": "La imagen ha sido analizada!", "predicted_class": predicted_class}
        col2.write(x)
