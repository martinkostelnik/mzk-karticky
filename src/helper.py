import re
from collections import defaultdict
import pandas as pd


LABELS = ["Author", "Title", "Original title", "Publisher", "Pages", "Series", "Edition", "References", "ID",
          "ISBN", "ISSN", "Topic", "Subtitle", "Date", "Institute", "Volume"]


class DatabaseRecordPattern:
    def __init__(self, regex=None, keys=None):
        self.regex = regex
        self.keys = keys

    def matches(self, key, text):
        if self._text_match(text) and self._key_match(key):
            return True

        return False

    def _text_match(self, text):
        if self.regex is None or self.regex is not None and re.match(self.regex, text):
            return True

        return False

    def _key_match(self, key):
        if self.keys is None or self.keys is not None and key in self.keys:
            return True

        return False


title_pattern = DatabaseRecordPattern(keys=["245", "24500", "24510"])

author_pattern = DatabaseRecordPattern(regex="^.*aut$", keys=["100", "700", "975", "978", "1000", "1001", "1002",
                                                              "1003", "1100", "7000", "7001", "7003", "9750", "9751",
                                                              "9811", "10010", "10013", "60017", "70002", "70010",
                                                              "70012", "71012", "1OR", "7001?", "100?", "700 7", "7OR"])

original_title_pattern = DatabaseRecordPattern(keys=["765", "7650"])

publisher_pattern = DatabaseRecordPattern(keys=["260"])

pages_pattern = DatabaseRecordPattern(keys=["300"])

series_pattern = DatabaseRecordPattern(keys=["490", "4901"])

edition_pattern = DatabaseRecordPattern(keys=["250"])

references_pattern = DatabaseRecordPattern(keys=["504"])

id_pattern = DatabaseRecordPattern(keys=["910"])

isbn_pattern = DatabaseRecordPattern(keys=["020"])

issn_pattern = DatabaseRecordPattern(keys=["022"])

topic_pattern = DatabaseRecordPattern(keys=["650"])

# patterns = [title_pattern, original_title_pattern, publisher_pattern, pages_pattern, series_pattern,
#             edition_pattern, references_pattern, id_pattern, isbn_pattern, issn_pattern]


def generate_db_records(db_key, text):
    parts = split_line(text)
    
    if author_pattern.matches(db_key, text):
        return {"Author": normalize_author(text)}

    if title_pattern.matches(db_key, text):
        return {"Title": parts["a"], "Subtitle": parts["b"]}

    if original_title_pattern.matches(db_key, text):
        return {"Original title": parts["t"]}

    if publisher_pattern.matches(db_key, text):
        return {"Publisher": f"{parts['a']}, {parts['b']}", "Date": parts["c"]}

    if pages_pattern.matches(db_key, text):
        return {"Pages": f"{parts['a']} {parts['e']}"}

    if series_pattern.matches(db_key, text):
        s = ""

        for key, val in parts.items():
            if key in ["v", "x"]:
                continue

            s += f"{val} "

        return {"Series": s, "Volume": parts["v"], "ISSN": parts["x"]}

    if edition_pattern.matches(db_key, text):
        return {"Edition": parts["a"]}

    if references_pattern.matches(db_key, text):
        return {"References": parts["a"]}

    if id_pattern.matches(db_key, text):
        s = ""

        IDs = text.split("$$")
        for id in IDs[1:]:
            s += f"{id[1:]} "

        return {"ID": s}

    if isbn_pattern.matches(db_key, text):
        return {"ISBN": parts["a"]}

    if issn_pattern.matches(db_key, text):
        return {"ISSN": parts["a"]}

    if topic_pattern.matches(db_key, text):
        return {"Topic": parts["a"]}

    return {}

def split_line(line):
    """ The '$$' is used as field separator in the individual DB records 
    """

    d = defaultdict(lambda: "")

    for part in line.split("$$")[1:]:
        d[part[0]] = part[1:]

    return d


def normalize_author(text):
    second_comma = text.find(',', text.find(',') + 1)
    if second_comma != -1:
        text = text[:second_comma]

    words = []
    for word in text.split():
        contains_number = any(['0' <= char <= '9' for char in word])
        is_aut = word == "aut"

        if contains_number or is_aut:
            break

        words.append(word)

    text = ' '.join(words)

    return text


def clean_db_record(in_path, out_path):
    """ This function takes file containing raw DB record and cleans it.
        Removes all lines not starting with correct ID and removes unneccessary columns.
        Stores the result in new file in format "<key> <content>"
    """

    lines = []

    with open(in_path, "r") as f:
        ID = in_path.rpartition("/")[2].partition(".")[0]
        lines = [f"{line[10:13]} {line[18:-1]}" for line in f if line.startswith(ID)]

    with open(out_path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")


def create_mapping(path):
    """ Loads file containing FILENAME to DB_ID mapping 
        { FILENAME: ID }
    """

    mapping = {}

    df = pd.read_csv(path, sep=" ", header=None, usecols=[0, 1], dtype={0: str, 1: str})
    
    for _, row in df.iterrows():
        # In the mapping file, '-' is used as folder separator, hence the replacement
        row[0] = row[0].replace("-", "/")
        mapping[row[0].partition(".")[0]] = f"{row[1]}"

    return mapping
