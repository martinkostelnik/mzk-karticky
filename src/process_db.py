import os

# This script expects a db record files to already be created.
# This can be done using the 'find_db_data()' function in 'create_tasks.py'
# It is dumb, to be fixed later, all db functionality should be in one script.
DB_FOLDER_IN = r"db"
DB_FOLDER_OUT = r"db-out"

MAPPING = {
    "100": "Author",
    "245": "Title",
    "765": "Original_Title",
    "260": "Publisher",
    "300": "Pages",
    "490": "Series",
    "250": "Edition",
    "504": "References",
    "910": "ID",
    "020": "ISBN",
    "022": "ISSN",
    "650": "Topic",
    "700": "Sec_Author",
}

INV_MAPPING = {v: k for k, v in MAPPING.items()}


def split_line(line):
    """ The '$$' is used as field separator in the individual DB records 
    """

    # res = {}

    # s = line.split("$$")

    # for part in s[1:]:
    #     res[part[0]] = part[1:]

    # return res
    return { part[0]: part[1:] for part in line.split("$$")[1:] }


def create_html_table(lines):
    """ HTML table can be imported into label-studio.
        We decided to do this for better readability.
    """

    table = ["<table>"]

    for line in lines:
        table += ["<tr>"]

        p_line = line.partition(" ")
        table += [f"<td>{p_line[0]}</td>"]
        table += [f"<td>{p_line[2]}</td>"]

        table += ["</tr>"]

    table += ["</table>"]

    return table


def process_file(filename):
    """ This function expects the DB record file to be in the raw format.
        TLDR: Just extracted from DB dump, no further changes.
    """
    path_in = f"{DB_FOLDER_IN}/{filename}"
    path_out = f"{DB_FOLDER_OUT}/{filename}"

    in_lines = []
    out_lines = []

    with open(path_in, "r") as f:
        in_lines = [f"{line[10:13]} {line[18:-1]}" for line in f if line.startswith(filename.partition(".")[0])]

    for line in in_lines:
        code = line.partition(" ")[0]
        parts = split_line(line)

        try:
            if code == INV_MAPPING["Author"]:
                out_lines.append(f"Author {parts['a']}\n")

            elif code == INV_MAPPING["Sec_Author"]:
                if "aut" in parts["4"]:
                    out_lines.append(f"Author_2 {parts['a']}\n")

            elif code == INV_MAPPING["Title"]:
                out_lines.append(f"Title {parts['a']}\n")
                out_lines.append(f"Subtitle {parts['b']}\n")

            elif code == INV_MAPPING["Original_Title"]:
                out_lines.append(f"Original_Title {parts['t']}\n")

            elif code == INV_MAPPING["Publisher"]:
                out_lines.append(f"Publisher {parts['a']}, {parts['b']}\n")
                out_lines.append(f"Date {parts['c']}\n")

            elif code == INV_MAPPING["Pages"]:
                out_lines.append(f"Pages {parts['a']}")
                out_lines[-1] = f"{out_lines[-1]} {parts['e']}"
                out_lines[-1] = f"{out_lines[-1]}\n"  

            elif code == INV_MAPPING["Series"]:
                s = f"Series "

                for key, val in parts.items():
                    if key in ["v", "x"]:
                        continue

                    s += f"{val} "

                out_lines.append(f"{s}\n")

                if "v" in parts:
                    out_lines.append(f"Volume {parts['v']}\n")

                if "x" in parts:
                    out_lines.append(f"ISSN {parts['x']}\n")

            elif code == INV_MAPPING["Edition"]:
                out_lines.append(f"Edition {parts['a']}\n")

            elif code == INV_MAPPING["References"]:
                out_lines.append(f"References {parts['a']}\n")

            elif code == INV_MAPPING["ID"]:
                IDs = line.split("$$")
                for id in IDs[1:]:
                    out_lines.append(f"ID {id[1:]}\n")

            elif code == INV_MAPPING["ISBN"]:
                out_lines.append(f"ISBN {parts['a']}\n")
            
            elif code == INV_MAPPING["ISSN"]:
                out_lines.append(f"ISSN {parts['a']}\n")

            elif code == INV_MAPPING["Topic"]:
                out_lines.append(f"Topic {parts['a']}\n")

        except KeyError:
            pass


    table = create_html_table(out_lines)
    output = table + ["<br>"] + [line + "<br>" for line in in_lines]

    with open(path_out, "w") as f:
        for line in output:
            f.write(line)


if __name__ == "__main__":
    for filename in os.listdir(DB_FOLDER_IN):
        process_file(filename)
