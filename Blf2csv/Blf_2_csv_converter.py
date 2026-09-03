"""
blf_to_csv.py

Reads a Vector BLF (Binary Logging Format) CAN log file and converts it
to a CSV file, with one row per CAN frame.

Requires:
    pip install python-can pandas --break-system-packages

Usage:
    Edit the CONFIG section below with your file paths, then run:
        python blf_to_csv.py
"""

import sys

import can
import pandas as pd

# ============================ CONFIG ============================
# Path to the input BLF file
BLF_FILE_PATH = r"C:\path\to\your\file.blf"

# Path to the output CSV file. None = same name as BLF file, .csv extension
CSV_FILE_PATH = None
# ==================================================================


def blf_to_dataframe(blf_path: str) -> pd.DataFrame:
    """Read all CAN messages from a BLF file into a pandas DataFrame."""
    rows = []
    try:
        reader = can.BLFReader(blf_path)
        for msg in reader:
            rows.append({
                "timestamp": msg.timestamp,
                "channel": msg.channel,
                "arbitration_id_hex": f"0x{msg.arbitration_id:X}",
                "arbitration_id_dec": msg.arbitration_id,
                "is_extended_id": msg.is_extended_id,
                "is_rx": not msg.is_rx == False,  # True unless explicitly TX
                "is_error_frame": msg.is_error_frame,
                "is_remote_frame": msg.is_remote_frame,
                "dlc": msg.dlc,
                "data_length": len(msg.data),
                "data_hex": msg.data.hex(sep=" ").upper(),
            })
    except Exception as exc:
        raise RuntimeError(f"Failed to read BLF file '{blf_path}': {exc}")

    if not rows:
        print("Warning: no messages were found in the BLF file.", file=sys.stderr)

    return pd.DataFrame(rows)


def convert(blf_path: str, csv_path: str = None) -> pd.DataFrame:
    df = blf_to_dataframe(blf_path)

    csv_path = csv_path or blf_path.rsplit(".", 1)[0] + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} messages to: {csv_path}")

    return df


def main():
    try:
        convert(BLF_FILE_PATH, CSV_FILE_PATH)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
