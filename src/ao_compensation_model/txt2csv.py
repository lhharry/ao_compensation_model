"""Convert text-delimited sensor files in a folder to semicolon-separated CSVs."""

import os

import pandas as pd


def convert_folder_to_csv(folder_path: str) -> None:
    """Convert all .txt and .csv files in a folder to semicolon-delimited CSVs.

    The delimiter of each file is auto-detected from the first line.
    Files are converted in-place (same name, .csv extension).

    :param folder_path: Path to the folder containing sensor files.
    """
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".txt", ".csv")):
            continue

        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()

            # Auto-detect delimiter
            if "," in first_line:
                delimiter = ","
            elif "\t" in first_line:
                delimiter = "\t"
            elif ";" in first_line:
                delimiter = ";"
            else:
                delimiter = ","

            df = pd.read_csv(
                file_path,
                sep=delimiter,
                engine="python",
                on_bad_lines="warn",
                encoding="utf-8",
                dtype=str,
            )
            df.columns = df.columns.str.strip()

            base_name = os.path.splitext(filename)[0]
            out_path = os.path.join(folder_path, f"{base_name}.csv")
            df.to_csv(out_path, index=False, sep=";")
            print(f"Converted: {filename} -> {base_name}.csv")

        except Exception as e:  # noqa: BLE001
            print(f"Could not process {filename}: {e}")

    print("All files converted.")


if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected = filedialog.askdirectory(
        title="Select Folder Containing Sensor Files",
        initialdir=os.path.dirname(os.path.abspath(__file__)),
    )

    if selected:
        convert_folder_to_csv(selected)
    else:
        print("Selection cancelled.")
