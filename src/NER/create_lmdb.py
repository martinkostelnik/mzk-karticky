import os
import lmdb
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-dir", help="Path to the root OCR dir.")
    parser.add_argument("--lmdb", help="Path to the output LMDB.")

    args = parser.parse_args()
    return args


def write(cur, key, value):
    cur.put(key.encode(), value.encode())


def load(path):
    with open(path) as file:
        text = file.read()

    return text


def main():
    args = parse_arguments()

    gb100 = 100000000000
    env = lmdb.open(args.lmdb, map_size=gb100)
    with env.begin(write=True) as txn:
        cur = txn.cursor()

        for root, dirs, files in os.walk(args.ocr_dir):
            for name in files:
                if not name.endswith(".txt"):
                    continue

                file_path = os.path.join(root, name)
                text = load(file_path)

                file_id = file_path.replace(args.ocr_dir, "")
                if file_id.startswith("/"):
                    file_id = file_id[1:]

                write(cur, file_id, text)

    return 0


if __name__ == "__main__":
    exit(main())
