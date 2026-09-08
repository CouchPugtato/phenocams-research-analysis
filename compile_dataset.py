import argparse
import os
import re
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

CAMERA = "NEON.D01.HOPB.DP1.20002"
PHENOCAM = "https://phenocam.nau.edu"
NEON_API = "https://data.neonscience.org/api/v0/data/query"
TIMEOUT = 120


def date(value):
    try:
        return datetime.strptime(value, "%m/%d/%Y")
    except ValueError:
        raise argparse.ArgumentTypeError("use MM/DD/YYYY")


def download_images(session, start, end, every, folder):
    url = f"{PHENOCAM}/data/archive/{CAMERA}/ROI/{CAMERA}-midday.txt"
    response = session.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    paths = response.text.splitlines()
    rows = []
    folder.mkdir(parents=True, exist_ok=True)

    for path in paths:
        match = re.search(r"_(\d{4}_\d{2}_\d{2}_\d{6})\.jpg$", path)
        if not match:
            continue
        stamp = datetime.strptime(match.group(1), "%Y_%m_%d_%H%M%S")
        if start <= stamp <= end + timedelta(days=1) and (stamp - start).days % every == 0:
            target = folder / Path(path).name
            if not target.exists():
                response = session.get(PHENOCAM + path, timeout=TIMEOUT)
                response.raise_for_status()
                target.write_bytes(response.content)
            utc_stamp = pd.Timestamp(stamp, tz=timezone(timedelta(hours=-5))).tz_convert("UTC")
            rows.append((stamp, utc_stamp, str(target)))

    return pd.DataFrame(rows, columns=["image_timestamp", "image_timestamp_utc", "image_file"])


def download_discharge(session, token, start, end):
    params = {
        "productCode": "DP4.00130.001", "siteCode": "HOPB",
        "startDateMonth": start.strftime("%Y-%m"), "endDateMonth": end.strftime("%Y-%m"),
        "package": "basic", "includeProvisional": "true",
    }
    response = session.get(NEON_API, params=params, headers={"X-API-Token": token}, timeout=TIMEOUT)
    response.raise_for_status()
    releases = response.json()["data"]["releases"]
    files = (f for release in releases for package in release["packages"] for f in package["files"])
    tables = []

    for file in files:
        name = file["name"]
        if "csd_15_min" in name and name.endswith(".csv"):
            columns = {"endDateTime": "neon_timestamp", "dischargeContinuous": "discharge"}
        elif "csd_continuousDischarge" in name and name.endswith(".csv"):
            columns = {"endDate": "neon_timestamp", "maxpostDischarge": "discharge"}
        else:
            continue
        csv = session.get(file["url"], timeout=TIMEOUT)
        csv.raise_for_status()
        table = pd.read_csv(StringIO(csv.text), usecols=columns).rename(columns=columns)
        table["source_table"] = "csd_15_min" if "csd_15_min" in name else "csd_continuousDischarge"
        tables.append(table)

    if not tables:
        raise RuntimeError("NEON returned no continuous-discharge tables for this period")
    data = pd.concat(tables, ignore_index=True)
    data["neon_timestamp"] = pd.to_datetime(data["neon_timestamp"], utc=True)
    data["discharge"] = pd.to_numeric(data["discharge"], errors="coerce")
    return data.dropna(subset=["neon_timestamp"]).sort_values("neon_timestamp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", type=date, help="MM/DD/YYYY")
    parser.add_argument("end_date", type=date, help="MM/DD/YYYY")
    parser.add_argument("interval", type=int, help="every N days")
    args = parser.parse_args()
    if args.start_date > args.end_date or args.interval < 1:
        parser.error("backwards dates or negative interval")
    token = os.getenv("NEON_token")
    if not token:
        parser.error("couldn't find NEON token in env")

    with requests.Session() as session:
        images = download_images(session, args.start_date, args.end_date, args.interval, Path("data/images"))
        if images.empty:
            parser.error("no PhenoCam images were found in that date range")
        discharge = download_discharge(session, token, args.start_date, args.end_date)

    result = pd.merge_asof(images.sort_values("image_timestamp_utc"), discharge,
                           left_on="image_timestamp_utc", right_on="neon_timestamp",
                           direction="nearest", tolerance=pd.Timedelta("1 day"))
    result = result.drop(columns=["image_timestamp_utc", "source_table"])
    Path("data").mkdir(exist_ok=True)
    result.to_csv("data/dataset.csv", index=False)
    print(f"Saved {len(result)} images and their nearest discharge values to data/dataset.csv")

if __name__ == "__main__":
    main()
