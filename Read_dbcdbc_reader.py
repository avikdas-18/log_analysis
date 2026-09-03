"""
dbc_to_dataframe.py

Reads a CAN DBC file and extracts message/signal details into pandas
DataFrames, which can be saved as CSV or Excel.

Requires:
    pip install cantools pandas openpyxl --break-system-packages

Usage:
    python dbc_to_dataframe.py path/to/file.dbc
    python dbc_to_dataframe.py path/to/file.dbc --format xlsx
"""

import argparse
import sys

import cantools
import pandas as pd


def load_dbc(dbc_path: str) -> cantools.database.can.Database:
    """Load a DBC file using cantools."""
    try:
        db = cantools.database.load_file(dbc_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse DBC file '{dbc_path}': {exc}")
    return db


def messages_to_dataframe(db: cantools.database.can.Database) -> pd.DataFrame:
    """One row per CAN message (frame)."""
    rows = []
    for msg in db.messages:
        rows.append({
            "message_name": msg.name,
            "frame_id_hex": f"0x{msg.frame_id:X}",
            "frame_id_dec": msg.frame_id,
            "is_extended_frame": msg.is_extended_frame,
            "length_bytes": msg.length,
            "sender": ", ".join(msg.senders) if msg.senders else None,
            "cycle_time_ms": msg.cycle_time,
            "comment": msg.comment,
            "num_signals": len(msg.signals),
        })
    return pd.DataFrame(rows)


def signals_to_dataframe(db: cantools.database.can.Database) -> pd.DataFrame:
    """One row per signal, with parent message info attached."""
    rows = []
    for msg in db.messages:
        for sig in msg.signals:
            rows.append({
                "message_name": msg.name,
                "frame_id_hex": f"0x{msg.frame_id:X}",
                "signal_name": sig.name,
                "start_bit": sig.start,
                "length_bits": sig.length,
                "byte_order": sig.byte_order,
                "is_signed": sig.is_signed,
                "is_float": sig.is_float,
                "factor": sig.scale,
                "offset": sig.offset,
                "minimum": sig.minimum,
                "maximum": sig.maximum,
                "unit": sig.unit,
                "receivers": ", ".join(sig.receivers) if sig.receivers else None,
                "initial_value": sig.initial,
                "comment": sig.comment,
                "choices": str(sig.choices) if sig.choices else None,
            })
    return pd.DataFrame(rows)


def nodes_to_dataframe(db: cantools.database.can.Database) -> pd.DataFrame:
    """One row per ECU/node defined in the DBC."""
    rows = [{"node_name": n.name, "comment": n.comment} for n in db.nodes]
    return pd.DataFrame(rows)


def export(dbc_path: str, out_format: str = "csv", out_prefix: str = None):
    db = load_dbc(dbc_path)

    out_prefix = out_prefix or dbc_path.rsplit(".", 1)[0]

    df_messages = messages_to_dataframe(db)
    df_signals = signals_to_dataframe(db)
    df_nodes = nodes_to_dataframe(db)

    if out_format == "csv":
        df_messages.to_csv(f"{out_prefix}_messages.csv", index=False)
        df_signals.to_csv(f"{out_prefix}_signals.csv", index=False)
        df_nodes.to_csv(f"{out_prefix}_nodes.csv", index=False)
        print(f"Wrote:\n  {out_prefix}_messages.csv\n  {out_prefix}_signals.csv\n  {out_prefix}_nodes.csv")
    elif out_format == "xlsx":
        out_file = f"{out_prefix}.xlsx"
        with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
            df_messages.to_excel(writer, sheet_name="Messages", index=False)
            df_signals.to_excel(writer, sheet_name="Signals", index=False)
            df_nodes.to_excel(writer, sheet_name="Nodes", index=False)
        print(f"Wrote: {out_file}")
    else:
        raise ValueError("out_format must be 'csv' or 'xlsx'")

    return df_messages, df_signals, df_nodes


def main():
    parser = argparse.ArgumentParser(description="Parse a DBC file into pandas DataFrames.")
    parser.add_argument("dbc_file", help="Path to the .dbc file")
    parser.add_argument("--format", choices=["csv", "xlsx"], default="csv",
                         help="Output format (default: csv)")
    parser.add_argument("--out-prefix", default=None,
                         help="Prefix/path for output file(s) (default: same name as DBC file)")
    args = parser.parse_args()

    try:
        export(args.dbc_file, out_format=args.format, out_prefix=args.out_prefix)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
