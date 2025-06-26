import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="embeddings/sentence-problem_type",
    repo_id="swardiantara/problem-type-embedding",
    repo_type="model",
)