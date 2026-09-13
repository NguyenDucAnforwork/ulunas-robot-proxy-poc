"""Upload a checkpoint/file to HF banhchungtuongot/ulunas-robot-proxy-poc.
Usage: python3 hf_upload.py <local_path> <repo_path>"""
import os, re, sys
from huggingface_hub import HfApi

def token():
    for line in open("/content/.env"):
        m = re.match(r'\s*HF_TOKEN\s*=\s*(.+?)\s*$', line)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    return None

if __name__ == "__main__":
    local, repo_path = sys.argv[1], sys.argv[2]
    api = HfApi(token=token())
    api.upload_file(path_or_fileobj=local, path_in_repo=repo_path,
                    repo_id="banhchungtuongot/ulunas-robot-proxy-poc", repo_type="model")
    print(f"uploaded {local} -> {repo_path}")
