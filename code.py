"""
CAN Log Analyzer
=================
Automates extraction of specific CAN signals from a "triplet-format" CSV log
(the common CANape/CANoe/vehicle-logger export where EACH signal has its own
Time[s] column, its own value column, and a blank spacer column, because
signals are transmitted asynchronously on the bus).

Header pattern (repeats every 3 columns):
    Time[s], <DBC>::<Message>::<SignalName>[unit], ,  Time[s], <next signal>, , ...

What this script does
----------------------
1. Reads the CSV (handles Latin-1 encoding, which is common for these exports).
2. Auto-locates each signal you ask for by matching its short name
   (e.g. "DCCurrentA") against the full column header
   (e.g. "G05 DBC_V1_0_latest::TM_MCU_STATUS_3::DCCurrentA[A]").
3. Extracts each signal as its own (time, value) series (dropping the blank
   padding rows that occur because different signals log at different rates).
4. Resamples every signal onto ONE common time grid at a fixed interval,
   using "hold last value" (forward-fill) - the standard way to resample CAN
   data, since a signal's value is valid until the next time it's transmitted.
5. Computes any calculated/derived signals (e.g. Generator power = I * V).
6. Writes a single tidy CSV: one row per timestamp, one column per signal.

Usage
-----
Set the two variables below (input file name and output time interval),
then just run:
    python3 can_log_analyzer.py

Customize the SIGNAL_MAP and CALCULATED_SIGNALS sections below for your project.
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. CONFIGURATION - edit this section for your project
# ---------------------------------------------------------------------------

# VARIABLE 1: input log file name (the CAN log CSV to read)
INPUT_LOG_FILE = "sample_data.csv"

# VARIABLE 2: time interval between two consecutive output rows, in seconds
# (this is the calibratable output logging rate, e.g. 1.0 = every 1 second,
# 0.1 = every 100ms, 5.0 = every 5 seconds, ...)
OUTPUT_TIME_INTERVAL = 1.0

# Output path: folder where the output CSV should be saved.
# Use "" (empty string) or "." to save in the current folder.
# Examples: "output", "/mnt/user-data/outputs", "C:/Users/me/Desktop/logs"
OUTPUT_FOLDER = ""

# Output file name (just the file name, not the full path)
OUTPUT_FILE_NAME = "resampled_output.csv"

# Map: output column name -> short signal name to search for in the CSV header.
# The search matches the text between the last "::" and an optional "[unit]"
# suffix, so you don't need to type the full DBC::Message::Signal[unit] path.
SIGNAL_MAP = {
    "DCCurrentA": "DCCurrentA",
    "DCVoltageV": "DCVoltageV",
    "MotorTorque": "MotorTorque",
    "Motorspeed": "Motorspeed",
    "GC_DC_VOLT": "GC_DC_VOLT",
    "GC_DC_CURR": "GC_DC_CURR",
    "GC_DC_CURR_PREC_IN": "GC_DC_CURR_PERC_IN",   # note: source file spells this PERC not PREC
    "GC_GENERATOR_SPEED": "GC_GENERATOR_SPEED",
    "Pack_Current": "Pack_Current",
    "Switched_Pack_Voltage": "Switched_Pack_Voltage",
    "Unswitched_Pack_Voltage": "Unswitched_Pack_Voltage",
    "Dbug_Avbl_Power_Dc_watt_out": "Dbug_Avbl_Power_Dc_watt_out",
    "Pressure_Proc": "Pressure_Proc",
    "Current_Feedback": "Current_Feedback",
}

# Calibration constants used in the power/efficiency calculations below.
Efficiency_of_hyd_to_mech = 0.92
Cal_EngineMax_trq = 90
Generator_Sys_efficiency = 0.81
# Calculated columns: name -> function that receives the resampled DataFrame
# and returns a new Series. Add as many as you like.
# NOTE: these run in order (top to bottom), and each function can reference
# any column added by a PREVIOUS entry in this dict (e.g. Current_Summation
# below uses the Battery_Charging_Status column computed just above it).
CALCULATED_SIGNALS = {
    "Generator_power": lambda df: df["GC_DC_CURR"] * df["GC_DC_VOLT"],

    # "Charging" if the generator DC bus voltage is higher than the pack
    # voltage (current flows into the pack), "Discharging" if lower.
    # When they're exactly equal, marked "Equal" (edge case not specified).
    "Battery_Charging_Status": lambda df: np.select(
        [df["GC_DC_VOLT"] > df["Unswitched_Pack_Voltage"],
         df["GC_DC_VOLT"] < df["Unswitched_Pack_Voltage"]],
        ["Charging", "Discharging"],
        default="Equal",
    ),

    # Current balance check: compares the generator/motor/pack currents
    # depending on which direction power is currently flowing.
    "Current_Summation": lambda df: np.select(
        [df["Battery_Charging_Status"] == "Charging",
         df["Battery_Charging_Status"] == "Discharging"],
        [df["GC_DC_CURR"] - (df["DCCurrentA"] + df["Pack_Current"]),
         df["DCCurrentA"] - (df["GC_DC_CURR"] + df["Pack_Current"])],
        default=np.nan,
    ),

    # Mechanical power delivered by the engine, from its max torque rating
    # and generator speed. (2*pi*T*N)/60 is the standard rotational power
    # formula, converting rpm to rad/s.
    "Mechanical_Power_output": lambda df: (
        2 * 3.141 * Cal_EngineMax_trq * df["GC_GENERATOR_SPEED"] * Efficiency_of_hyd_to_mech
    ) / 60,

    # Hydraulic pump/motor displacement, derived from the measured current
    # feedback (mA) via a linear calibration: (I - 400) * 0.125.
    "Displacement_in_cc": lambda df: (df["Current_Feedback"] - 400) * 0.125,

    # Hydraulic power delivered mechanically, from displacement, pressure,
    # and generator speed.
    "Hydraulic_power_to_mechanical": lambda df: (
        df["Displacement_in_cc"] * df["Pressure_Proc"] * df["GC_GENERATOR_SPEED"]
    ) / 600,

    # Actual hydraulic power output, correcting for hydraulic-to-mechanical
    # conversion losses.
    "Actual_hydraulic_power_output": lambda df: (
        df["Hydraulic_power_to_mechanical"] / Efficiency_of_hyd_to_mech
    ),

    # Power left over for the generator after hydraulic power is subtracted
    # from total mechanical power, scaled by the generator system efficiency.
    "Calculated_available_power_for_generator": lambda df: (
        (df["Mechanical_Power_output"] - df["Actual_hydraulic_power_output"]) * Generator_Sys_efficiency
    ),
}


# ---------------------------------------------------------------------------
# 2. CORE LOGIC - generally no need to edit below this line
# ---------------------------------------------------------------------------

def detect_encoding_and_read_header(path):
    """These logger exports are frequently Latin-1 / Windows-1252, not UTF-8."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, encoding=enc) as f:
                header = f.readline().strip().split(",")
            return enc, header
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path} with utf-8/latin-1/cp1252")


def find_signal_column(header, short_name):
    """
    Locate the value-column index for a signal given its short name.
    Matches "...::<short_name>" or "...::<short_name>[unit]".
    Returns the column index of the VALUE column (time column is index-1).
    """
    for i, col in enumerate(header):
        if col.endswith("::" + short_name) or ("::" + short_name + "[") in col:
            return i
    return None


def extract_signal_series(df, value_col_idx):
    """
    Given the raw (headerless) dataframe and a value column index, return a
    clean (time, value) pandas Series, dropping the blank padding rows.
    """
    time_col_idx = value_col_idx - 1
    t = pd.to_numeric(df[time_col_idx], errors="coerce")
    v = pd.to_numeric(df[value_col_idx], errors="coerce")
    mask = t.notna() & v.notna()
    t, v = t[mask], v[mask]
    # Guard against duplicate/out-of-order timestamps
    series = pd.Series(v.values, index=t.values).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series


def load_signals(csv_path, signal_map):
    """Read the CSV and extract each requested signal as a raw time series."""
    encoding, header = detect_encoding_and_read_header(csv_path)
    print(f"Detected encoding: {encoding}")

    raw = pd.read_csv(csv_path, encoding=encoding, header=None, skiprows=1, low_memory=False)

    series_dict = {}
    missing = []
    for out_name, short_name in signal_map.items():
        idx = find_signal_column(header, short_name)
        if idx is None:
            missing.append((out_name, short_name))
            continue
        s = extract_signal_series(raw, idx)
        series_dict[out_name] = s
        print(f"  Found '{out_name}' (matched '{short_name}') -> "
              f"{len(s)} samples, t=[{s.index.min():.3f}, {s.index.max():.3f}]s")

    if missing:
        print("\nWARNING: the following signals were NOT found in the CSV header:")
        for out_name, short_name in missing:
            print(f"  - {out_name} (searched for '{short_name}')")

    return series_dict


def resample_to_common_grid(series_dict, interval):
    """
    Build one common time grid spanning the union of all signals' time ranges,
    then hold-last-value (forward-fill) each signal onto that grid.
    """
    if not series_dict:
        raise RuntimeError("No signals were successfully extracted - nothing to resample.")

    t_min = min(s.index.min() for s in series_dict.values())
    t_max = max(s.index.max() for s in series_dict.values())
    grid = np.arange(t_min, t_max + interval, interval)

    out = pd.DataFrame(index=grid)
    out.index.name = "Time[s]"

    for name, s in series_dict.items():
        # reindex onto the union of existing timestamps + grid, ffill, then
        # sample exactly at the grid points (this correctly "holds" the last
        # transmitted value rather than interpolating between transitions)
        combined_index = s.index.union(grid)
        held = s.reindex(combined_index).ffill()
        out[name] = held.reindex(grid).values

    return out


def add_calculated_signals(df, calculated_signals):
    for name, fn in calculated_signals.items():
        try:
            df[name] = fn(df)
        except KeyError as e:
            print(f"WARNING: could not compute '{name}' - missing input signal {e}")
    return df


def main():
    print(f"Loading signals from {INPUT_LOG_FILE} ...")
    series_dict = load_signals(INPUT_LOG_FILE, SIGNAL_MAP)

    print(f"\nResampling to a common {OUTPUT_TIME_INTERVAL}s grid (hold-last-value)...")
    df = resample_to_common_grid(series_dict, OUTPUT_TIME_INTERVAL)

    print("Computing calculated signals...")
    df = add_calculated_signals(df, CALCULATED_SIGNALS)

    # nice column order: requested signals in original order, then calculated ones
    ordered_cols = [c for c in SIGNAL_MAP.keys() if c in df.columns]
    ordered_cols += [c for c in CALCULATED_SIGNALS.keys() if c in df.columns]
    df = df[ordered_cols]

    # Build the full output path, creating the destination folder if needed
    if OUTPUT_FOLDER and OUTPUT_FOLDER != ".":
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE_NAME)
    else:
        output_path = OUTPUT_FILE_NAME

    df.to_csv(output_path, float_format="%.4f")
    print(f"\nDone. Wrote {len(df)} rows x {len(df.columns)} columns to {output_path}")


if __name__ == "__main__":
    main()
