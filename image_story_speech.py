# import os
# import requests
# from transformers import pipeline
# from langchain_groq import ChatGroq
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from dotenv import load_dotenv
# from PIL import Image

# load_dotenv()

# api_key = os.getenv('GROQ_API_KEY')


# # Image 2 text
# def image_to_text(image_path):
#     try:
#         pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
#         image = Image.open(image_path)
#         text = pipe(image)
#         print("Image to Text:", text[0]["generated_text"])
#         return text[0]["generated_text"]
#     except Exception as e:
#         print(f"An error occurred: {e}")

# # LLM to generate story
# def generate_story(scenario):
#     template = """
#     You are an amazing storyteller.
#     You can generate a short story based on a simple narrative. The story should be no more than 50 words.

#     CONTEXT: {scenario}
#     STORY:
#     """
#     prompt = PromptTemplate(template=template, input_variables=["scenario"])

#     story_llm = LLMChain(
#         llm=ChatGroq(api_key=api_key, model="llama3-70b-8192", temperature=0.5),
#         prompt=prompt,
#         verbose=True
#     )

#     story = story_llm.predict(scenario=scenario)
#     print("Generated Story:", story)
#     return story

# # Text to speech
# def text2speech(message, output_path='audio.wav'):
#     API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-eng"
#     headers = {"Authorization": f"Bearer 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"}
#     payload = {"inputs": message}
#     response = requests.post(API_URL, headers=headers, json=payload)
    
#     if response.status_code == 200:
#         with open(output_path, 'wb') as f:
#             f.write(response.content)
#         print(f"Audio saved to {output_path}")
#     else:
#         print(f"An error occurred: {response.status_code}, {response.text}")

# def main(image_path):
#     # Step 1: Convert image to text
#     image_text = image_to_text(image_path)
#     if not image_text:
#         return

#     # Step 2: Generate a story based on the text from the image
#     story = generate_story(image_text)
#     if not story:
#         return

#     # Step 3: Convert the generated story to speech
#     text2speech(story)

# if __name__ == "__main__":
#     main("t.jpg")


# Streamlit UI Code 

import os
import requests
from transformers import pipeline
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from PIL import Image
import streamlit as st

# Load environment variables
load_dotenv()

api_key = os.getenv('GROQ_API_KEY')

# Image 2 text function
def image_to_text(image_path):
    try:
        pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        image = Image.open(image_path)
        text = pipe(image)
        return text[0]["generated_text"]
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# LLM to generate story
def generate_story(scenario):
    template = """
    You are an amazing storyteller.
    You can generate a short story based on a simple narrative. The story should be no more than 50 words.

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(
        llm=ChatGroq(api_key=api_key, model="llama3-70b-8192", temperature=0.5),
        prompt=prompt,
        verbose=True
    )

    try:
        story = story_llm.predict(scenario=scenario)
        return story
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Text to speech
def text2speech(message, output_path='audio.wav'):
    API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-eng"
    headers = {"Authorization": f"Bearer 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"}
    payload = {"inputs": message}
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return output_path
    else:
        st.error(f"An error occurred: {response.status_code}, {response.text}")
        return None

# Streamlit app
def main():
    st.set_page_config(
        page_title="Image to Story and Audio",
        page_icon=":camera:",
        layout="wide"
    )

    st.title("Image to Story and Audio using Hugging Face")

    with st.expander("Project Summary"):
        st.write("This project takes an image as input, generates a caption, creates a short story based on the caption, and converts the story into an audio file using Hugging Face models.")

    with st.sidebar:
        st.header(":computer: Tech Stack")
        st.write("• Hugging Face Transformers (image-to-text, language models)")
        st.write("• LangChain (LLMChain, ChatGroq)")
        st.write("• Streamlit (UI framework)")
        st.write("• PIL (image processing)")
        st.write("• Requests (API calls)")

    uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Step 1: Convert image to text
        with st.spinner("Generating caption..."):
            image_text = image_to_text(uploaded_image)
        
        if image_text:
            st.subheader("Image Caption")
            st.write(image_text)

            # Step 2: Generate a story based on the text from the image
            with st.spinner("Generating story..."):
                story = generate_story(image_text)
            
            if story:
                st.subheader("Generated Story")
                st.write(story)

                # Step 3: Convert the generated story to speech
                with st.spinner("Converting story to speech..."):
                    audio_path = text2speech(story)
                
                if audio_path:
                    st.subheader("Generated Audio")
                    audio_file = open(audio_path, "rb")
                    st.audio(audio_file.read(), format="audio/wav")
                    st.download_button("Download Audio", audio_path, "story_audio.wav")

if __name__ == "__main__":
    main()