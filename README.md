# Image to Story and Audio using Hugging Face

## Project Overview

This project is designed to take an image as input, generate a caption from the image, create a short story based on the caption, and then convert the story into an audio file. The application is built using Streamlit for the user interface and leverages various AI models from Hugging Face for the tasks of image captioning, story generation, and text-to-speech conversion.

## Features

- **Image Captioning**: Uses Hugging Face's `Salesforce/blip-image-captioning-base` model to generate a caption for the uploaded image.
- **Story Generation**: Utilizes LangChain's `ChatGroq` model to generate a short story based on the caption.
- **Text-to-Speech**: Converts the generated story into an audio file using Hugging Face's `facebook/mms-tts-eng` model.
- **Streamlit UI**: Provides an easy-to-use interface for uploading images, viewing the generated caption and story, and listening to or downloading the audio file.

## Tech Stack

- **Hugging Face Transformers**: For image-to-text and language models.
- **LangChain**: For LLMChain and ChatGroq integration.
- **Streamlit**: For the user interface.
- **PIL**: For image processing.
- **Requests**: For API calls.
- **Python Dotenv**: For loading environment variables.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/AbhishekSharma-17/HuggingFace.git
    cd HuggingFace
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the Required Packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables**:
    Create a `.env` file in the project directory and add your API keys:
    ```env
    GROQ_API_KEY=your_groq_api_key
    HUGGINGFACE_API_KEY=your_huggingface_api_key
    ```

## Usage

1. **Run the Streamlit Application**:
    ```bash
    streamlit run app.py
    ```

2. **Open the Application**:
    Open the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.

3. **Upload an Image**:
    - Click on the "Upload an Image" button and select an image file (png, jpg, or jpeg).

4. **View the Generated Caption**:
    - The app will display the generated caption for the uploaded image.

5. **Generate a Story**:
    - The app will create a short story based on the caption and display it.

6. **Convert Story to Audio**:
    - The app will convert the story to an audio file. You can listen to it directly in the app or download it.

## File Structure

- `app.py`: The main Streamlit application file.
- `requirements.txt`: List of required Python packages.
- `.env`: Environment variables file (not included in the repository, needs to be created).
- `README.md`: This README file.

## Acknowledgements

- Hugging Face for providing state-of-the-art models.
- LangChain for the LLMChain and ChatGroq integration.
- Streamlit for the user-friendly UI framework.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Feel free to contribute to this project by submitting issues or pull requests. Happy coding!
