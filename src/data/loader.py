"""
Data loader module for SEGMENTIQ.

Responsible solely for loading and validating the presence of raw dataset files
and expected schema columns without applying transformations or cleaning.
"""

from pathlib import Path
from typing import Any, Sequence, Union
import pandas as pd

# Project root directory (3 levels up from src/data/loader.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default project-relative path to raw dataset
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail.xlsx"

# Expected schema columns in the raw dataset
EXPECTED_COLUMNS: tuple[str, ...] = (
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
)


def load_raw_data(
    file_path: Union[str, Path, None] = None,
    sheet_name: Union[str, int] = 0,
    expected_columns: Sequence[str] = EXPECTED_COLUMNS,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load the raw Online Retail Excel dataset into a Pandas DataFrame and validate schema.

    This function resolves file paths relative to the project root when a relative path
    or no path is provided. It performs file existence verification and validates
    that all expected columns are present.

    Note:
        This function strictly performs data loading and structural validation.
        It does NOT clean data, remove cancellations/duplicates, filter invalid values,
        impute missing CustomerIDs, or compute any RFM metrics.

    Parameters:
        file_path (str | Path | None):
            Path to the .xlsx dataset. If None, defaults to 'data/raw/online_retail.xlsx'
            relative to the project root.
        sheet_name (str | int):
            Sheet name or index to read from the Excel workbook. Defaults to 0 (first sheet).
        expected_columns (Sequence[str]):
            List or tuple of column names required to exist in the dataset.
            Defaults to EXPECTED_COLUMNS.
        **kwargs (Any):
            Additional keyword arguments forwarded directly to `pd.read_excel()`
            (e.g., `nrows=100`, `engine="openpyxl"`).

    Returns:
        pd.DataFrame:
            Raw dataset as loaded from the Excel file with expected columns verified.

    Raises:
        FileNotFoundError:
            If the specified or resolved file does not exist on disk.
        ValueError:
            If any of the expected columns are missing from the loaded dataset.
    """
    if file_path is None:
        target_path = DEFAULT_DATA_PATH
    else:
        path_obj = Path(file_path)
        if path_obj.is_absolute():
            target_path = path_obj
        else:
            # Check if path is directly resolvable from current working dir,
            # otherwise resolve relative to project root
            if path_obj.exists():
                target_path = path_obj.resolve()
            else:
                target_path = (PROJECT_ROOT / path_obj).resolve()

    if not target_path.exists() or not target_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset file not found at: '{target_path}'. "
            f"Please ensure the file exists at the expected project path."
        )

    df = pd.read_excel(target_path, sheet_name=sheet_name, **kwargs)

    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset loaded from '{target_path}' is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


# Alias for backward compatibility or alternate naming convention
load_data = load_raw_data

__all__ = [
    "load_raw_data",
    "load_data",
    "EXPECTED_COLUMNS",
    "DEFAULT_DATA_PATH",
    "PROJECT_ROOT",
]
