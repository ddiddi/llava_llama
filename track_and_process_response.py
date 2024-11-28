import json
from llama_cpp import Llama
import re

def extract_objects_from_response(response_text):
    """
    Parses the response text to identify potential objects mentioned in the image description.
    """
    # Simple heuristic: Identify nouns or key objects in the description
    objects = re.findall(r'\b(?:field|boardwalk|grass|sky|clouds|atmosphere)\b', response_text, re.IGNORECASE)
    return list(set(objects))  # Return unique objects

def process_response_and_calculate_odd(response):
    """
    Processes the response to extract objects and calculate Object Density Descriptor (ODD).
    """
    # Extract the description content
    description = response.get("message", {}).get("content", "")
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

def run_llama_with_json(input_data):
    """
    Runs LLama with JSON schema to provide a summary of object counts and ODD.
    """
    llm = Llama(model_path="Llama-3.2-1B-Instruct-Q4_K_M.gguf", chat_format="chatml")
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs in JSON.",
            },
            {
                "role": "user",
                "content": f"Based on the data: {input_data}, provide the object count and ODD.",
            },
        ],
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "object_count": {"type": "integer"},
                    "odd": {"type": "number"},
                    "details": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["object_count", "odd"],
            },
        },
        temperature=0.7,
    )
    return response

def main():
    # Simulated response from the previous LLama execution
    llama_response = {
        'index': 0,
        'message': {
            'role': 'assistant',
            'content': (
                ' The image depicts a serene, grassy field with a wooden boardwalk running through it. '
                'The boardwalk is surrounded by tall grass, creating a natural, peaceful atmosphere. '
                'The sky above is a beautiful blue, dotted with a few clouds. The scene is captured from '
                'a high vantage point, providing a comprehensive view of the field and the boardwalk.'
            )
        },
        'logprobs': None,
        'finish_reason': 'stop'
    }

    # Process the response to calculate object count and ODD
    processed_data = process_response_and_calculate_odd(llama_response)

    print("Processed Data:")
    print(json.dumps(processed_data, indent=4))

    # Run LLama with JSON schema
    llama_json_response = run_llama_with_json(processed_data)
    print("LLama JSON response:")
    print(json.dumps(llama_json_response, indent=4))

if __name__ == "__main__":
    main()
