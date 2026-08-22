import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image

st.set_page_config(page_title="Text to Image Generator")

st.title("🎨 Text to Image Generator")

@st.cache_resource
def load_model():
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")
    return pipe

prompt = st.text_input(
    "Enter your prompt",
    placeholder="A beautiful sunset over a mountain lake"
)

if st.button("Generate Image"):
    if prompt.strip():

        with st.spinner("Loading model and generating image..."):
            try:
                pipe = load_model()

                image = pipe(
                    prompt=prompt,
                    num_inference_steps=1,
                    guidance_scale=0.0
                ).images[0]

                st.image(
                    image,
                    caption="Generated Image",
                    use_container_width=True
                )

            except Exception as e:
                st.error("Something went wrong while generating the image.")
                st.exception(e)

    else:
        st.warning("Please enter a prompt.")