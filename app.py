import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image

st.set_page_config(
    page_title="AI Image Generation",
    page_icon="🖼️"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }

    h1 {
        color: white;
        text-align: center;
    }

    p, label {
        color: white !important;
    }

    .stButton > button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🖼️ AI Image Generation")

st.write(
    "✨ Describe the image you want, and AI will generate it!"
)

prompt = st.text_area(
    "✍️ Describe your image:",
    "A cute cat sitting in a beautiful garden at sunset",
    height=100
)


@st.cache_resource
def load_model():

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float32
    )

    pipe = pipe.to("cpu")

    return pipe


if st.button("🎨 Generate Image"):

    if prompt.strip() == "":
        st.warning("Please enter a description.")

    else:

        with st.spinner("⏳ Loading AI model..."):

            pipe = load_model()

        with st.spinner("🎨 Generating your image..."):

            image = pipe(
                prompt=prompt,
                num_inference_steps=1,
                guidance_scale=0.0
            ).images[0]

        st.success("🎉 Image generated successfully!")

        st.image(
            image,
            caption="✨ Your AI Generated Image",
            use_container_width=True
        )