""" Creates tasks in JSON format for label-studio import.
"""

import pandas as pd
import os
import shutil


PREFIX = r"/mnt/matylda5/ibenes/projects/pero/MZK-karticky/"
DATA_FOLDER = r"mzk_karticky"
MAPPING_FILE = r"files.txt"
TASK_FOLDER = r"tasks"
IMAGE_FOLDER = r"images"
OCR_FOLDER = r"page-txts"
DB_FOLDER = r"db"


def create_mapping():
    """ Loads file containing FILENAME to DB_ID mapping
    """

    mapping = {}

    df = pd.read_csv(MAPPING_FILE, sep=" ", header=None, usecols=[0, 1], dtype={0: str, 1: str})
    
    for _, row in df.iterrows():
        # In the mapping file, '-' is used as folder separator, hence the replacement
        row[0] = row[0].replace("-", "/")
        mapping[row[0].partition(".")[0]] = f"{row[1]}"

    return mapping


def copy_files(mapping):
    """ Helper function to copy all mapped scans and transcripts
    """

    for key, _ in mapping.items():
        shutil.copy2(f"{PREFIX}all-karticky-png/{key}.gif.png", f"./{IMAGE_FOLDER}")
        shutil.copy2(f"{PREFIX}all-karticky-ocr/{key}.gif.xml.txt", f"./{OCR_FOLDER}/")


def find_db_data(mapping):
    """ Helper function to find FILENAME to DB_ID mappings in DB dump
    """
    
    for _, value in mapping.items():
        os.system(f"cat mzk01.bib mzk03.bib | grep {value} > {DB_FOLDER}/{value}.txt")


def create_tasks(mapping):
    id = 1

    for key, value in mapping.items():
        filename = key.rpartition("/")[2]

        with open(f"{TASK_FOLDER}/task{id}.json", "x") as task_file:
            task_file.write("{\n") # Begin file

            task_file.write(f'  "id": {id},\n')
            task_file.write('  "data": {\n') # Begin "data"

            task_file.write(f'          "text": "/data/local-files/?d={DATA_FOLDER}/{OCR_FOLDER}/{filename}.gif.xml.txt",\n') # OCR DATA
            task_file.write(f'          "image": "/data/local-files/?d={DATA_FOLDER}/{IMAGE_FOLDER}/{filename}.gif.png",\n') # IMAGE DATA
            task_file.write(f'          "db": "/data/local-files/?d={DATA_FOLDER}/{DB_FOLDER}/{value}.txt"\n') # DB DATA

            task_file.write("  }\n") # End "data"
            task_file.write("}\n") # End file
        
        id += 1


if __name__ == "__main__":
    mapping = create_mapping()
    #copy_files(mapping)
    #find_db_data(mapping)
    #create_tasks(mapping)
