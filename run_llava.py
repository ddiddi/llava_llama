from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler
import base64

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

# Initialize the model and chat handler
chat_handler = MoondreamChatHandler.from_pretrained(
  repo_id="vikhyatk/moondream2",
  filename="*mmproj*",
)

llm = Llama.from_pretrained(
  repo_id="vikhyatk/moondream2",
  filename="*text-model*",
  chat_handler=chat_handler,
  n_ctx=2048, # n_ctx should be increased to accommodate the image embedding
)

# Example 1: URL-based image
response = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
            ]
        }
    ]
)
print("Response for URL-based image:")
print(response["choices"][0])

# Example 2: Local image
# file_path = 'file_path.png'  # Replace with your actual local image path
# data_uri = image_to_base64_data_uri(file_path)

# messages = [
#     {"role": "system", "content": "You are an assistant who perfectly describes images."},
#     {
#         "role": "user",
#         "content": [
#             {"type": "image_url", "image_url": {"url": data_uri}},
#             {"type": "text", "text": "Describe this image in detail please."}
#         ]
#     }
# ]

# response_local = llm.create_chat_completion(messages=messages)
# print("Response for Local Image:")
# print(response_local["choices"][0]["text"])
