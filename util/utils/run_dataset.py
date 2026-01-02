#!/usr/bin/env python3
import argparse
import os
import re
import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
YEAR_FILE_RE = re.compile(r"flight_with_weather_(\d{4})\.csv$")


def default_input_dir() -> Path:
    candidate = ROOT_DIR / "data/Aeolus/Flight_Tab"
    return candidate if candidate.is_dir() else Path.cwd()


@contextmanager
def pushd(new_dir: Path):
    prev_dir = Path.cwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)


def discover_year_files(input_dir: Path):
    year_files = []
    for path in sorted(input_dir.glob("flight_with_weather_*.csv")):
        match = YEAR_FILE_RE.match(path.name)
        if match:
            year_files.append((int(match.group(1)), path))
    return year_files


def select_year_files(input_dir: Path, years):
    year_files = discover_year_files(input_dir)
    if not year_files:
        return []
    if not years:
        return year_files

    year_set = set(years)
    filtered = [item for item in year_files if item[0] in year_set]
    missing = sorted(year_set - {year for year, _ in filtered})
    if missing:
        print(f"Warning: missing year files: {', '.join(map(str, missing))}")
    return filtered


def remove_outliers_percentile(df, column, lower=0.01, upper=0.99):
    lower_bound = df[column].quantile(lower)
    upper_bound = df[column].quantile(upper)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def extract_data(input_csv: Path, output_csv: Path, start_date: str, end_date: str):
    df = pd.read_csv(input_csv, low_memory=False)
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df = df.dropna(subset=["FL_DATE"])

    mask = (df["FL_DATE"] >= start_date) & (df["FL_DATE"] <= end_date)
    filtered_df = df.loc[mask]
    filtered_df.to_csv(output_csv, index=False)
    print(f"Saved {len(filtered_df)} rows to {output_csv}")


def process_filtered_data(input_csv: Path, output_csv: Path, info_yaml: Path):
    df = pd.read_csv(input_csv, low_memory=False)
    if "ARR_DELAY" not in df.columns:
        raise ValueError("ARR_DELAY column is required.")

    df = remove_outliers_percentile(df, "ARR_DELAY")

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df = df.dropna(subset=["FL_DATE"])
    df["FL_YEAR"] = df["FL_DATE"].dt.year
    df.rename(
        columns={"MONTH": "FL_MONTH", "DAY_OF_MONTH": "FL_DAY", "DAY_OF_WEEK": "FL_WEEK"},
        inplace=True,
    )
    df.drop(columns=["FL_DATE"], inplace=True)

    time_columns = [
        "CRS_DEP_TIME",
        "DEP_TIME",
        "WHEELS_OFF",
        "WHEELS_ON",
        "CRS_ARR_TIME",
        "ARR_TIME",
    ]
    for col in time_columns:
        dt = pd.to_datetime(df[col], errors="coerce")
        df[col + "_MIN"] = dt.dt.hour * 60 + dt.dt.minute
    df.drop(columns=time_columns, inplace=True)

    encoder_columns = ["OP_CARRIER", "ORIGIN", "DEST"]
    encoder = LabelEncoder()
    for col in encoder_columns:
        if col in df.columns:
            df[col] = encoder.fit_transform(df[col].astype(str))

    if "ORIGIN_INDEX" not in df.columns and "ORIGIN" in df.columns:
        df["ORIGIN_INDEX"] = df["ORIGIN"]
    if "DEST_INDEX" not in df.columns and "DEST" in df.columns:
        df["DEST_INDEX"] = df["DEST"]

    categorical_columns = [
        "OP_CARRIER",
        "OP_CARRIER_FL_NUM",
        "FL_YEAR",
        "FL_MONTH",
        "FL_DAY",
        "FL_WEEK",
        "ORIGIN_INDEX",
        "DEST_INDEX",
    ]

    continuous_columns = [
        "CRS_DEP_TIME_MIN",
        "CRS_ARR_TIME_MIN",
        "CRS_ELAPSED_TIME",
        "FLIGHTS",
        "O_TEMP",
        "O_PRCP",
        "O_WSPD",
        "D_TEMP",
        "D_PRCP",
        "D_WSPD",
        "O_LATITUDE",
        "O_LONGITUDE",
        "D_LATITUDE",
        "D_LONGITUDE",
    ]

    target = ["DEP_DELAY", "ARR_DELAY"]

    required = target + categorical_columns + continuous_columns
    missing = [col for col in required if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")

    scaler = MinMaxScaler()
    df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

    df = df[target + categorical_columns + continuous_columns]
    df.to_csv(output_csv, index=False)

    info_buf = StringIO()
    df.info(buf=info_buf)
    data_info = {
        "columns_info": {
            "Target": target,
            "Categorical Features": categorical_columns,
            "Continuous Features": continuous_columns,
        },
        "data_summary": {
            "info": info_buf.getvalue(),
            "memory_usage": df.memory_usage(deep=True).to_dict(),
        },
    }

    with open(info_yaml, "w") as yaml_file:
        yaml.safe_dump(data_info, yaml_file, default_flow_style=False)

    print(f"Saved processed data to {output_csv}")
    print(f"Saved data info to {info_yaml}")


def run_tab(input_dir: Path, output_dir: Path, years):
    from Datasets.Flight_tab import process_file, save_year_data

    year_files = select_year_files(input_dir, years)
    if not year_files:
        print(f"No year files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for year, path in year_files:
        print(f"Processing tabular data for {year}...")
        df = process_file(str(path))
        if df is None:
            print(f"Skipping {year}: processing returned None.")
            continue
        save_year_data(df, year, str(output_dir))


def run_chain(input_dir: Path, years):
    from Datasets import Flight_chain

    year_files = select_year_files(input_dir, years)
    if not year_files:
        print(f"No year files found in {input_dir}")
        return

    with pushd(input_dir):
        for year, _ in year_files:
            Flight_chain.process_year_data(year)


def run_network(input_dir: Path, years):
    from Datasets import Flight_networks

    year_files = select_year_files(input_dir, years)
    if not year_files:
        print(f"No year files found in {input_dir}")
        return

    for _, path in year_files:
        Flight_networks.process_year_file(str(path))


def build_parser():
    parser = argparse.ArgumentParser(description="Dataset processing entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Filter data by date range.")
    extract.add_argument("--input", required=True, type=Path, help="Path to source CSV.")
    extract.add_argument(
        "--output", default=Path("filtered_flight_data.csv"), type=Path, help="Output CSV path."
    )
    extract.add_argument("--start-date", required=True, help="Start date, e.g. 2024-06-01.")
    extract.add_argument("--end-date", required=True, help="End date, e.g. 2024-06-15.")

    pro = subparsers.add_parser("pro", help="Process filtered data into tabular features.")
    pro.add_argument(
        "--input", default=Path("results/filtered_flight_data.csv"), type=Path, help="Input CSV path."
    )
    pro.add_argument(
        "--output", default=Path("results/arr_delay_data.csv"), type=Path, help="Output CSV path."
    )
    pro.add_argument(
        "--info",
        default=Path("results/arr_delay_data_info.yaml"),
        type=Path,
        help="Output YAML path.",
    )

    tab = subparsers.add_parser("tab", help="Build per-year tabular datasets.")
    tab.add_argument(
        "--input-dir",
        default=default_input_dir(),
        type=Path,
        help="Directory with flight_with_weather_YYYY.csv files.",
    )
    tab.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Output directory for tabular datasets.",
    )
    tab.add_argument("--years", nargs="*", type=int, help="Years to process.")

    chain = subparsers.add_parser("chain", help="Build flight chain datasets.")
    chain.add_argument(
        "--input-dir",
        default=default_input_dir(),
        type=Path,
        help="Directory with flight_with_weather_YYYY.csv files.",
    )
    chain.add_argument("--years", nargs="*", type=int, help="Years to process.")

    network = subparsers.add_parser("network", help="Build flight network graphs.")
    network.add_argument(
        "--input-dir",
        default=default_input_dir(),
        type=Path,
        help="Directory with flight_with_weather_YYYY.csv files.",
    )
    network.add_argument("--years", nargs="*", type=int, help="Years to process.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        extract_data(args.input, args.output, args.start_date, args.end_date)
        return

    if args.command == "pro":
        process_filtered_data(args.input, args.output, args.info)
        return

    if args.command == "tab":
        input_dir = args.input_dir
        output_dir = args.output_dir or (input_dir / "Tab")
        run_tab(input_dir, output_dir, args.years)
        return

    if args.command == "chain":
        run_chain(args.input_dir, args.years)
        return

    if args.command == "network":
        run_network(args.input_dir, args.years)
        return


if __name__ == "__main__":
    main()
