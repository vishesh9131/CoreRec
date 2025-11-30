import ast
import itertools
import xml.etree.ElementTree as ET
import json
import struct
import csv
from collections import Counter


def see_ui_parser(tokens, line_idx, id_inline=False, **kwargs):
    if id_inline:
        return [(str(line_idx + 1), iid, 1.0) for iid in tokens]
    else:
        return [(tokens[0], iid, 1.0) for iid in tokens[1:]]


def see_uir_parser(tokens, **kwargs):
    return [(tokens[0], tokens[1], float(tokens[2]))]


def see_uirt_parser(tokens, **kwargs):
    return [(tokens[0], tokens[1], float(tokens[2]), int(tokens[3]))]


# ... other parser functions similar to data.py ...

SEE_PARSERS = {
    "UI": see_ui_parser,
    "UIR": see_uir_parser,
    "UIRT": see_uirt_parser,
    # ... add other parsers as needed ...
}


class SeeReader:
    """Reader class for reading data with different types of format in see.py.

    Parameters
    ----------
    user_set: set, default = None
        Set of users to be retained when reading data.
        If `None`, all users will be included.

    item_set: set, default = None
        Set of items to be retained when reading data.
        If `None`, all items will be included.

    # ... other parameters similar to data.py ...
    """

    def __init__(
        self,
        user_set=None,
        item_set=None,
        min_user_freq=1,
        min_item_freq=1,
        # ... other parameters ...
        encoding="utf-8",
        errors=None,
    ):
        self.user_set = (
            user_set if (user_set is None or isinstance(user_set, set)) else set(user_set)
        )
        self.item_set = (
            item_set if (item_set is None or isinstance(item_set, set)) else set(item_set)
        )
        self.min_uf = min_user_freq
        self.min_if = min_item_freq
        # ... initialize other attributes ...
        self.encoding = encoding
        self.errors = errors

    def _filter(self, tuples, fmt="UIR"):
        # ... similar filtering logic as in data.py ...
        return tuples

    def see(self, fpath, fmt="UIR", sep="\t", skip_lines=0, id_inline=False, parser=None, **kwargs):
        """Read data and parse line by line based on provided `fmt` or `parser`.

        Parameters
        ----------
        fpath: str
            Path to the data file.

        fmt: str, default: 'UIR'
            Line format to be parsed ('UI', 'UIR', 'UIRT', etc.)

        sep: str, default: '\t'
            The delimiter string.

        skip_lines: int, default: 0
            Number of first lines to skip

        id_inline: bool, default: False
            If `True`, user ids corresponding to the line numbers of the file,
            where all the ids in each line are item ids.

        parser: function, default: None
            Function takes a list of `str` tokenized by `sep` and
            returns a list of tuples which will be joined to the final results.
            If `None`, parser will be determined based on `fmt`.

        Returns
        -------
        tuples: list
            Data in the form of list of tuples. What inside each tuple
            depends on `parser` or `fmt`.

        """
        parser = SEE_PARSERS.get(fmt, None) if parser is None else parser
        if parser is None:
            raise ValueError(
                "Invalid line format: {}\n" "Supported formats: {}".format(fmt, SEE_PARSERS.keys())
            )

        file_extension = fpath.split(".")[-1].lower()

        if file_extension in ["csv", "xls"]:
            data = self._read_tabular(fpath, sep)
        elif file_extension == "xml":
            data = self._read_xml(fpath)
        elif file_extension == "txt":
            data = self._read_text(fpath)
        elif file_extension == "json":
            data = self._read_json(fpath)
        elif file_extension in ["md", "txt"]:
            data = self._read_text(fpath)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        tuples = [
            tup
            for idx, line in enumerate(data)
            for tup in parser(line.strip().split(sep), line_idx=idx, id_inline=id_inline, **kwargs)
        ]
        tuples = self._filter(tuples=tuples, fmt=fmt)
        return tuples

    def _read_tabular(self, fpath, sep):
        """Read CSV or XLS files."""
        with open(fpath, encoding=self.encoding, errors=self.errors) as f:
            return [line.strip() for line in f if line.strip()]

    def _read_csv(self, fpath):
        """Read CSV files."""
        with open(fpath, encoding=self.encoding, errors=self.errors) as f:
            return [line.strip() for line in f if line.strip()]

    def _read_xml(self, fpath):
        """Read XML files."""
        tree = ET.parse(fpath)
        root = tree.getroot()
        return [ET.tostring(element, encoding="unicode") for element in root]

    def _read_json(self, fpath):
        """Read JSON files."""
        with open(fpath, encoding=self.encoding, errors=self.errors) as f:
            return json.load(f)

    def _read_text(self, fpath):
        """Read text or markdown files."""
        with open(fpath, encoding=self.encoding, errors=self.errors) as f:
            return f.readlines()

    def read_list(self, data_list):
        """Read data from a list."""
        return data_list

    def read_dict(self, data_dict):
        """Read data from a dictionary."""
        return list(data_dict.items())


# Conversion functions
def convert_csv_to_vish(csv_filename, vish_filename):
    with open(csv_filename, "r") as f, open(vish_filename, "wb") as vish_file:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row

        # Write a dummy header (will be updated later)
        vish_file.write(struct.pack("4sIII", b"VISH", 0, 0, 0))
        count = 0

        for row in reader:
            if len(row) < 5:
                print(f"Skipping row with insufficient columns: {row}")
                continue

            try:
                user_id, item_id, action, rating, timestamp = map(int, row[:4]) + [float(row[4])]
                vish_file.write(
                    struct.pack("Q Q B f I", user_id, item_id, action, rating, timestamp)
                )
                count += 1
            except ValueError as e:
                print(f"Skipping row with invalid data: {row}. Error: {e}")

        # Update header with actual counts
        vish_file.seek(0)
        vish_file.write(struct.pack("4sIII", b"VISH", count, 0, 0))


def convert_json_to_vish(json_filename, vish_filename):
    with open(json_filename, "r") as f, open(vish_filename, "wb") as vish_file:
        data = json.load(f)
        users, items, interactions = data["users"], data["items"], data["interactions"]

        # Write header
        vish_file.write(struct.pack("4sIII", b"VISH", len(users), len(items), len(interactions)))

        # Write users
        for user in users:
            vish_file.write(
                struct.pack(
                    "Q B 7s", user["id"], user["age"], user["preferences"].encode("utf-8")[:7]
                )
            )

        # Write items
        for item in items:
            vish_file.write(
                struct.pack(
                    "Q H H 4s",
                    item["id"],
                    item["category"],
                    item["year"],
                    item["attributes"].encode("utf-8")[:4],
                )
            )

        # Write interactions
        for interaction in interactions:
            vish_file.write(
                struct.pack(
                    "Q Q B f I",
                    interaction["user_id"],
                    interaction["item_id"],
                    interaction["action"],
                    interaction["rating"],
                    interaction["timestamp"],
                )
            )


def convert_xml_to_vish(xml_filename, vish_filename):
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    interactions = root.findall(".//interaction")

    with open(vish_filename, "wb") as vish_file:
        vish_file.write(struct.pack("4sIII", b"VISH", 0, 0, len(interactions)))

        for interaction in interactions:
            user_id = int(interaction.find("user_id").text)
            item_id = int(interaction.find("item_id").text)
            action = int(interaction.find("action").text)
            rating = float(interaction.find("rating").text)
            timestamp = int(interaction.find("timestamp").text)
            vish_file.write(struct.pack("Q Q B f I", user_id, item_id, action, rating, timestamp))


def convert_to_vish(file_path, vish_filename, file_type):
    if file_type == "csv":
        convert_csv_to_vish(file_path, vish_filename)
    elif file_type == "json":
        convert_json_to_vish(file_path, vish_filename)
    elif file_type == "xml":
        convert_xml_to_vish(file_path, vish_filename)
    else:
        raise ValueError("Unsupported file format")
    print(f"Converted {file_path} to {vish_filename}")


if __name__ == "__main__":
    # Example usage of SeeReader to read a text file
    reader = SeeReader()

    # Path to the text file
    text_file_path = "corerec/data/ex.csv"

    data = reader.see(fpath=text_file_path, fmt="UIRT", sep=",")
    for entry in data:
        print(entry)

    # Example usage of conversion functions
    # convert_to_vish("corerec/data/ex.csv", "corerec/data/first.vish", "csv")
    # convert_to_vish("dataset.json", "dataset.vish", "json")
    # convert_to_vish("dataset.xml", "dataset.vish", "xml")
