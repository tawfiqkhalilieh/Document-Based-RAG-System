from huggingface_hub import snapshot_download


def main() -> None:
    """
    Tiny helper script to download Qwen3-4B-Instruct-2507 locally.
    """
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    local_dir = "qwen3-4b-instruct-2507"

    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision="main",
    )
    print(f"Model downloaded to: {local_dir}")


if __name__ == "__main__":
    main()

