import os
import argparse
import hashlib
import tarfile
import urllib.request

VOC_URL = "https://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
VOC_TAR = "VOCtrainval_11-May-2012.tar"
VOC_MD5 = "6cd6e144f989b92b3379bac3b3de84fd"  # 官方常见md5

# 没用到这个file, 手动下载

def md5sum(path, chunk_size=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download(url, out_path):
    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    print("Download finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--skip_md5", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    tar_path = os.path.join(args.root, VOC_TAR)
    target_dir = os.path.join(args.root, "VOCdevkit", "VOC2012")

    if os.path.isdir(target_dir):
        print(f"VOC2012 already exists at: {target_dir}")
        return

    if not os.path.exists(tar_path):
        download(VOC_URL, tar_path)
    else:
        print(f"Found existing tar: {tar_path}")

    size = os.path.getsize(tar_path)
    print(f"Tar size: {size / (1024**2):.2f} MB")
    if size < 100 * 1024 * 1024:
        raise RuntimeError(
            "Downloaded file is too small (likely an HTML error page). "
            "Please check network/proxy and try again."
        )

    if not args.skip_md5:
        print("Checking md5...")
        m = md5sum(tar_path)
        print("md5 =", m)
        if m != VOC_MD5:
            raise RuntimeError(
                f"MD5 mismatch: expected {VOC_MD5}, got {m}. "
                "Delete the tar and retry."
            )

    print("Extracting...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=args.root)
    print(f"Done. Extracted to: {target_dir}")


if __name__ == "__main__":
    main()
