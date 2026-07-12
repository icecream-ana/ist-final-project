import json
import time
import urllib.request
from pathlib import Path

ann_path = Path("data/annotations/instances_val2017.json")
train_dir = Path("data/train")
val_dir = Path("data/val")

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

with ann_path.open("r", encoding="utf-8") as f:
    images = json.load(f)["images"]

tasks = []
for im in images[:500]:
    tasks.append(("train", im["file_name"], train_dir / im["file_name"]))

for im in images[500:600]:
    tasks.append(("val", im["file_name"], val_dir / im["file_name"]))

failed = []

def download_one(file_name, dst, retries=5):
    urls = [
        "https://images.cocodataset.org/val2017/" + file_name,
        "http://images.cocodataset.org/val2017/" + file_name,
    ]

    if dst.exists() and dst.stat().st_size > 0:
        return True

    tmp = dst.with_suffix(dst.suffix + ".part")

    for attempt in range(1, retries + 1):
        for url in urls:
            try:
                print(f"Downloading {file_name} attempt={attempt} url={url}")
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                tmp.write_bytes(data)
                tmp.replace(dst)
                return True
            except Exception as e:
                print(f"Failed {file_name}: {e}")
                time.sleep(2 * attempt)

    if tmp.exists():
        tmp.unlink()
    return False

for i, split, file_name, dst in [(i, *task) for i, task in enumerate(tasks, 1)]:
    print(f"[{i}/{len(tasks)}] {split}: {file_name}")
    ok = download_one(file_name, dst)
    if not ok:
        failed.append((split, file_name))

print("Done.")
print("train count:", len(list(train_dir.glob("*.jpg"))))
print("val count:", len(list(val_dir.glob("*.jpg"))))
print("failed count:", len(failed))

if failed:
    Path("data/failed_downloads.txt").write_text(
        "\n".join(f"{split}\t{name}" for split, name in failed),
        encoding="utf-8",
    )
    print("Failed list saved to data/failed_downloads.txt")
