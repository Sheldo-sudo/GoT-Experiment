import argparse
from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError


DEFAULT_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "tokenizer*",
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
]


def download_qwen(
    repo_id: str = "Qwen/Qwen2.5-3B-Instruct",
    local_dir: str = "./models/Qwen2.5-3B-Instruct",
    revision: str = "main",
) -> str:
    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            revision=revision,
            local_dir_use_symlinks=False,
            allow_patterns=DEFAULT_ALLOW_PATTERNS,
        )
        return path
    except (RepositoryNotFoundError, RevisionNotFoundError) as e:
        suggestions = [
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen3-4B-Instruct-2507",
            "Qwen/Qwen3-4B",
        ]
        message = (
            f"Failed to download repo_id='{repo_id}' revision='{revision}'.\n"
            f"Original error: {e}\n\n"
            "Try one of these repo IDs:\n- "
            + "\n- ".join(suggestions)
            + "\n\nIf repo is gated/private, run `huggingface-cli login` first."
        )
        raise RuntimeError(message) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Qwen safetensors model snapshot.")
    parser.add_argument("--repo_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--local_dir", type=str, default="./models/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", type=str, default="main")
    args = parser.parse_args()

    out = download_qwen(args.repo_id, args.local_dir, args.revision)
    print(f"Downloaded model to: {out}")
