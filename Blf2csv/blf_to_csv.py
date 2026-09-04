"""
BLF to CSV Converter
====================
Converts Vector BLF (Binary Logging Format) CAN log files to CSV,
decoding raw messages using DBC (CAN Database) files.

Output format matches the sample_data.csv structure:
  - Each signal gets 3 columns: Time[s], <DBC>::<Message>::<Signal>[unit], <empty>
  - Rows are organized by message reception timestamp
"""

import os
import sys
import glob
import csv
import cantools
import can
from collections import OrderedDict


def load_all_dbc_files(dbc_folder):
    """
    Load all DBC files from the given folder.
    Returns a dict mapping: arbitration_id -> (db, dbc_basename, message)
    for quick lookup when decoding BLF messages.
    """
    dbc_files = glob.glob(os.path.join(dbc_folder, "*.dbc"))
    if not dbc_files:
        print(f"ERROR: No DBC files found in '{dbc_folder}'")
        sys.exit(1)

    # Map: arbitration_id -> (db, dbc_basename_without_ext, message_obj)
    id_to_msg = {}
    # Also keep track of all signal column definitions in DBC order
    all_signal_columns = []

    for dbc_path in sorted(dbc_files):
        dbc_basename = os.path.splitext(os.path.basename(dbc_path))[0]
        print(f"  Loading DBC: {dbc_basename}")
        try:
            db = cantools.database.load_file(dbc_path)
        except Exception as e:
            print(f"  WARNING: Failed to load {dbc_path}: {e}")
            continue

        for msg in db.messages:
            # Skip the VECTOR__INDEPENDENT_SIG_MSG (id 0xC0000000 / 3221225472)
            # These are placeholder signals not actually transmitted on the bus.
            if msg.frame_id == 0xC0000000 or msg.name == "VECTOR__INDEPENDENT_SIG_MSG":
                continue

            # Determine if this message uses extended ID
            # CAN IDs > 0x7FF are extended (29-bit), standard is 11-bit
            arb_id = msg.frame_id
            is_extended = msg.is_extended_id if hasattr(msg, 'is_extended_id') else (arb_id > 0x7FF)

            # Store with extended flag for proper matching
            key = (arb_id, is_extended)
            id_to_msg[key] = (db, dbc_basename, msg)

            # Build signal column list for this message
            for signal in msg.signals:
                unit = signal.unit if signal.unit else ""
                if unit and unit != "NA" and unit != "0":
                    col_name = f"{dbc_basename}::{msg.name}::{signal.name}[{unit}]"
                else:
                    col_name = f"{dbc_basename}::{msg.name}::{signal.name}"
                all_signal_columns.append({
                    "col_name": col_name,
                    "dbc": dbc_basename,
                    "msg_name": msg.name,
                    "sig_name": signal.name,
                    "msg_id": arb_id,
                    "is_extended": is_extended,
                })

    print(f"  Total messages mapped: {len(id_to_msg)}")
    print(f"  Total signal columns: {len(all_signal_columns)}")
    return id_to_msg, all_signal_columns


def convert_blf_to_csv(blf_path, dbc_folder, output_path):
    """
    Convert a single BLF file to CSV.

    Strategy:
    1. Load all DBC files and build signal column map.
    2. Read BLF file, decode every message using DBC definitions.
    3. Group decoded values by a time-based row concept:
       - Each row represents one "scan" of all messages at a point in time.
       - We collect all messages and their timestamps, then build rows
         matching the sample_data.csv format.
    """
    print(f"\nConverting: {os.path.basename(blf_path)}")
    print(f"  Loading DBC files from: {dbc_folder}")

    id_to_msg, all_signal_columns = load_all_dbc_files(dbc_folder)

    if not all_signal_columns:
        print("  ERROR: No signals found in DBC files. Aborting.")
        return

    # ---- Phase 1: Read BLF and decode all messages ----
    print(f"  Reading BLF file: {blf_path}")
    # Collect all decoded events as list of (timestamp, msg_key, decoded_signals)
    decoded_events = []
    msg_count = 0
    decoded_count = 0
    skipped_ids = set()

    try:
        with can.BLFReader(blf_path) as reader:
            for msg in reader:
                msg_count += 1
                if msg_count % 100000 == 0:
                    print(f"    Processed {msg_count} messages...")

                key = (msg.arbitration_id, msg.is_extended_id)
                if key not in id_to_msg:
                    skipped_ids.add(msg.arbitration_id)
                    continue

                db, dbc_basename, db_msg = id_to_msg[key]

                try:
                    decoded = db.decode_message(
                        db_msg.frame_id,
                        msg.data,
                        decode_choices=False
                    )
                    decoded_events.append((msg.timestamp, key, decoded))
                    decoded_count += 1
                except Exception:
                    # Silently skip messages that fail to decode (e.g. DLC mismatch)
                    pass

    except Exception as e:
        print(f"  ERROR reading BLF: {e}")
        return

    print(f"  Total CAN messages read: {msg_count}")
    print(f"  Successfully decoded: {decoded_count}")
    if skipped_ids:
        print(f"  Skipped {len(skipped_ids)} unknown arbitration IDs")

    if not decoded_events:
        print("  WARNING: No messages were decoded. CSV will be empty.")
        return

    # ---- Phase 2: Build time-aligned rows ----
    # Strategy: Group consecutive messages into rows.
    # A new row starts when we see the same message ID again (cycle restart)
    # or when a sufficient time gap occurs.

    print("  Building CSV rows...")

    # Sort events by timestamp
    decoded_events.sort(key=lambda x: x[0])

    # Build a mapping from signal column name to index for fast lookup
    sig_col_index = {}
    for i, sc in enumerate(all_signal_columns):
        sig_col_index[sc["col_name"]] = i

    # Build lookup: (msg_id, is_extended, sig_name) -> column index
    sig_lookup = {}
    for i, sc in enumerate(all_signal_columns):
        lookup_key = (sc["msg_id"], sc["is_extended"], sc["sig_name"])
        sig_lookup[lookup_key] = i

    # Group events into rows.
    # Each row contains: for each signal column, (timestamp, value) or (None, None)
    rows = []
    current_row = [None] * len(all_signal_columns)  # Each entry: (timestamp, value) or None
    seen_msg_ids_in_row = set()

    # We use the concept: collect until we see a message ID that was already in the current row,
    # which signals the start of a new cycle. Also start a new row on significant time gap.
    FIRST_TS = decoded_events[0][0]

    for timestamp, msg_key, decoded_signals in decoded_events:
        msg_id, is_extended = msg_key

        # Check if this message ID was already seen in this row → new row
        if msg_id in seen_msg_ids_in_row:
            rows.append(current_row)
            current_row = [None] * len(all_signal_columns)
            seen_msg_ids_in_row = set()

        seen_msg_ids_in_row.add(msg_id)

        # Fill in signal values for this message
        for sig_name, sig_value in decoded_signals.items():
            lookup_key = (msg_id, is_extended, sig_name)
            if lookup_key in sig_lookup:
                col_idx = sig_lookup[lookup_key]
                rel_timestamp = round(timestamp - FIRST_TS, 6)
                current_row[col_idx] = (rel_timestamp, sig_value)

    # Don't forget the last row
    if any(v is not None for v in current_row):
        rows.append(current_row)

    print(f"  Total rows generated: {len(rows)}")

    # ---- Phase 3: Write CSV ----
    print(f"  Writing CSV: {output_path}")

    # Build header: for each signal, 3 columns: Time[s], signal_name, empty
    header = []
    for sc in all_signal_columns:
        header.append("Time[s]")
        header.append(sc["col_name"])
        header.append("")  # empty separator column

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in rows:
            csv_row = []
            for entry in row:
                if entry is not None:
                    ts, val = entry
                    csv_row.append(ts)
                    csv_row.append(val)
                    csv_row.append("")  # separator
                else:
                    csv_row.append("")
                    csv_row.append("")
                    csv_row.append("")
            writer.writerow(csv_row)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  CSV written successfully! Size: {file_size_mb:.2f} MB")


def main():
    # Paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blf_folder = os.path.join(script_dir, "blf_files")
    dbc_folder = os.path.join(script_dir, "dbc_files")
    csv_folder = os.path.join(script_dir, "csv_files")

    print("=" * 60)
    print("BLF to CSV Converter")
    print("=" * 60)
    print(f"BLF folder: {blf_folder}")
    print(f"DBC folder: {dbc_folder}")
    print(f"CSV output: {csv_folder}")

    # Find all BLF files
    blf_files = glob.glob(os.path.join(blf_folder, "*.blf"))
    if not blf_files:
        print(f"\nERROR: No BLF files found in '{blf_folder}'")
        sys.exit(1)

    print(f"\nFound {len(blf_files)} BLF file(s) to convert:")
    for f in blf_files:
        print(f"  - {os.path.basename(f)}")

    # Convert each BLF file
    for blf_path in blf_files:
        blf_name = os.path.splitext(os.path.basename(blf_path))[0]
        output_path = os.path.join(csv_folder, f"{blf_name}.csv")
        convert_blf_to_csv(blf_path, dbc_folder, output_path)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
