from pathlib import Path

RAW = Path("data")
OUT = Path("outputs")
START, END = "2006-03", "2026-09"
HORIZONS = range(25)

SERIES = {
    "diesel": "GASDESM",
    "fafh": "CUSR0000SEFV",
    "core": "CPILFESL",
    "wage": "CES7000000003",
    "dspi": "DSPIC96",
    "ppi_food": "WPUID61111",
    "wage_all": "CES0500000003",
}
GAPPED = ["fafh", "core"]
COVID = [("2020-03", "2021-12")]
OCT25 = [("2025-10", "2025-10")]