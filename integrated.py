import streamlit as st
import base64
import json
import re
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler

@st.cache_resource
def load_model():
    # Initialize the model and chat handler
    chat_handler = MoondreamChatHandler.from_pretrained(
        repo_id="vikhyatk/moondream2",
        filename="*mmproj*",
    )

    llm = Llama.from_pretrained(
        repo_id="vikhyatk/moondream2",
        filename="*text-model*",
        chat_handler=chat_handler,
        n_ctx=2048,  # Increase context to accommodate the image embedding
    )
    return llm

def extract_objects_from_response(response_text):
    """
    Parses the response text to identify potential objects mentioned in the image description.
    """
    # Define a list of common objects to search for
    common_objects = [
        'people', 'person', 'man', 'woman', 'child', 'boy', 'girl',
        'car', 'vehicle', 'truck', 'bus', 'bicycle', 'motorcycle',
        'plane', 'airplane', 'jet', 'helicopter',
        'dog', 'cat', 'animal', 'bird', 'horse', 'cow', 'sheep', 'elephant',
        'tree', 'flower', 'grass', 'forest', 'mountain', 'beach', 'ocean',
        'building', 'house', 'skyscraper', 'bridge', 'road', 'street',
        'sky', 'cloud', 'sun', 'moon', 'star',
        'boat', 'ship', 'train',
        'food', 'fruit', 'vegetable', 'drink',
        'computer', 'phone', 'camera',
        # Add more objects as needed
    ]

    # Create a regex pattern from the list of common objects
    pattern = r'\b(?:' + '|'.join(re.escape(obj) for obj in common_objects) + r')\b'
    
    # Find all occurrences of the common objects in the response text
    objects_found = re.findall(pattern, response_text, re.IGNORECASE)
    return list(set(objects_found))  # Return unique objects

def process_response_and_calculate_odd(response_text):
    """
    Processes the response to extract objects and calculate Object Density Descriptor (ODD).
    """
    # Extract the description content
    description = response_text
    objects = extract_objects_from_response(description)
    object_count = len(objects)

    # Example area for ODD calculation (e.g., a predefined constant or based on context)
    area = 100  # Replace with the actual area context if available
    odd = object_count / area if area else 0

    return {
        "description": description,
        "object_count": object_count,
        "odd": odd,
        "objects": objects,
    }

def run_llama_with_json(input_data, llm):
    """
    Runs LLama to provide a summary of object counts and ODD in JSON format.
    """
    prompt = (
        "You are a helpful assistant that outputs in JSON.\n"
        f"Based on the data: {json.dumps(input_data)}, provide the object count and ODD in JSON format.\n"
        "The JSON should have the following structure:\n"
        "{\n"
        '  "object_count": integer,\n'
        '  "odd": number,\n'
        '  "details": [list of strings]\n'
        "}\n"
        "Please output only the JSON."
    )
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs in JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    return response

def main():
    st.title("Image Processing with Llama")

    # Provide options for the user to either capture an image or upload one
    option = st.selectbox(
        "How would you like to provide the image?",
        ("Capture from Webcam", "Upload from Device")
    )

    img_bytes = None

    if option == "Capture from Webcam":
        # Use st.camera_input to capture image from webcam
        captured_image = st.camera_input("Take a picture")

        if captured_image is not None:
            # Read the image bytes
            img_bytes = captured_image.getvalue()
    else:
        # Use st.file_uploader to upload an image
        uploaded_image = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

        if uploaded_image is not None:
            # Read the image bytes
            img_bytes = uploaded_image.read()

    if img_bytes is not None:
        # Convert image to base64 data URI
        base64_data = base64.b64encode(img_bytes).decode('utf-8')
        data_uri = f"data:image/png;base64,{base64_data}"

        # Prepare the messages for the Llama model
        messages = [
            {"role": "system", "content": "You are an assistant who perfectly describes images."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "Describe this image in detail please."}
                ]
            }
        ]

        # Load the Llama model
        llm = load_model()

        # Run the model and display the response
        with st.spinner('Processing...'):
            response = llm.create_chat_completion(messages=messages)

        # Extract the assistant's description
        description_text = response["choices"][0]["message"]["content"]

        # Process the response to calculate object count and ODD
        processed_data = process_response_and_calculate_odd(description_text)

        # Run LLama with JSON schema
        llama_json_response = run_llama_with_json(processed_data, llm)
        json_response_text = llama_json_response["choices"][0]["message"]["content"]
        try:
            json_response = json.loads(json_response_text)
        except json.JSONDecodeError:
            json_response = {"error": "Failed to parse JSON"}

        # Create two columns with adjusted widths if necessary
        col1, col2 = st.columns([1, 1])

        # Display the captured or uploaded image in the first column
        with col1:
            st.header("Provided Image")
            st.image(img_bytes, caption='Provided Image', use_container_width=True)

        # Display the assistant's description and processed data in the second column
        with col2:
            st.header("Assistant's Description")
            st.write(description_text)
            st.header("Processed Data")
            st.write("Objects Detected:", processed_data["objects"])
            st.write("Object Count:", processed_data["object_count"])
            st.write("ODD:", processed_data["odd"])

            st.header("LLama JSON Response")
            st.write(json_response)

    else:
        st.write("Please provide an image to proceed.")

if __name__ == "__main__":
    main()
