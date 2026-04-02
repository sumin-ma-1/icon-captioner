import os, json

SRC_DIR = "data/processed/icon_caption_jsonl_gt"
FILES = ["train.jsonl", "val.jsonl"]

OLD_PREFIX = "/Users/suminma/Desktop/Research/dl:paper/icon_vlm/data/merged/images/train"

# 로컬 실제 이미지 폴더로 (절대경로)
NEW_PREFIX = "/Users/suminma/Desktop/Research/dl_paper/icon_vlm/data/merged/images/train"

def patch_file(path: str):
    out_path = path + ".patched"
    n = 0
    with open(path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            obj = json.loads(line)
            p = obj["image_path"]
            if p.startswith(OLD_PREFIX):
                obj["image_path"] = p.replace(OLD_PREFIX, NEW_PREFIX, 1)
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    os.replace(out_path, path)
    print(f"[OK] patched {path} ({n} lines)")

def main():
    for fn in FILES:
        p = os.path.join(SRC_DIR, fn)
        assert os.path.isfile(p), f"missing: {p}"
        patch_file(p)

if __name__ == "__main__":
    main()