"""
Usage:
    python3 policy_downloader.py
    python3 policy_downloader.py --output-dir /custom/path/policy
"""

import argparse
from pathlib import Path 

HF_REPO_ID = "Siddarth09/Hanuman_mars"
POLICY_FILES = ["hanuman_policy.onnx","hanuman_policy.onnx.data"]

def download_policy(out_dir: Path) -> Path:

    onnx_path = out_dir/"hanuman_policy.onnx"
    data_path = out_dir/"hanuman_policy.onnx.data"

    if onnx_path.exists() and data_path.exists():
        print(f"Policy already exists in {out_dir}")
        print(f"  {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")
        print(f"  {data_path} ({data_path.stat().st_size / 1024:.1f} KB)")
        return onnx_path
    
    print(f"Downloading policy from HuggingFace:{HF_REPO_ID}")
    out_dir.mkdir(parents=True,exist_ok=True)

    try: 
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("pip install hf")


    for filename in POLICY_FILES:
        print(f"{filename}..")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir= str(out_dir),
            repo_type="model"

        )

    print(f"\n✓ Policy downloaded to: {out_dir}")
    print(f"  {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")
    print(f"  {data_path} ({data_path.stat().st_size / 1024:.1f} KB)")
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Download HANUMAN policy from hugging face")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent/"policy"),
        help= "Directory to save policy files"
    )
    args= parser.parse_args()
    download_policy(Path(args.output_dir))

if __name__ == "__main__":
    main()