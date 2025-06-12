import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="./embedding",
    repo_id="swardiantara/message-problem-embedding",
    repo_type="model",
)

# api.upload_folder(
#     folder_path="./num_3_one_5_0.5-two_2_0.05/stage_two_model",
#     repo_id="swardiantara/sentence-problem-embedding",
#     repo_type="model",
# )