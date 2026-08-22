import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image

st.title("Text to Image Generator")

@st.cache_resource
def load_model():
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")
    return pipe

prompt = st.text_input("Enter your prompt")

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating image..."):
            pipe = load_model()

            image = pipe(
                prompt=prompt,
                num_inference_steps=1,
                guidance_scale=0.0
            ).images[0]

            st.image(image, caption="Generated Image")
    else:
        st.warning("Please enter a prompt.")