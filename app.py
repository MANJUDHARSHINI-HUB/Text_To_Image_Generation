import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image

st.set_page_config(
    page_title="AI Image Generation",
    page_icon="🖼️",
    layout="centered"
)

# -----------------------------
# Page design
# -----------------------------

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

# -----------------------------
# Title
# -----------------------------

st.title("🖼️ AI Image Generation")

st.write(
    "✨ Describe the image you want, and AI will generate it!"
)

# -----------------------------
# Prompt
# -----------------------------

prompt = st.text_area(
    "✍️ Describe your image:",
    "A cute cat sitting in a beautiful garden at sunset",
    height=100
)

# -----------------------------
# Load SD-Turbo
# -----------------------------

@st.cache_resource
def load_model():

    model_path = r"D:\huggingface_cache\hub\models--stabilityai--sd-turbo\snapshots\b261bac6fd2cf515557d5d0707481eafa0485ec2"

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        local_files_only=True
    )

    pipe = pipe.to("cpu")

    return pipe


# -----------------------------
# Generate image
# -----------------------------

if st.button("🎨 Generate Image"):

    if prompt.strip() == "":
        st.warning("Please enter a description.")

    else:

        try:

            with st.spinner("⏳ Loading SD-Turbo..."):

                pipe = load_model()

            st.success("✅ SD-Turbo loaded successfully!")

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

        except Exception as e:

            st.error("❌ Image generation failed.")

            st.exception(e)