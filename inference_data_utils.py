import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


SUPPORTED_DATA_FORMATS = {"xlsx", "csv", "jsonl"}


def resolve_data_format(data_path: str, input_format: str) -> str:
    """Resolve data format from explicit flag or file suffix.

    Args:
        data_path: Input dataset path.
        input_format: User-provided format, or "auto".

    Returns:
        str: Resolved format in {"xlsx", "csv", "jsonl"}.
    """
    if input_format != "auto":
        if input_format not in SUPPORTED_DATA_FORMATS:
            raise ValueError(f"Unsupported input_format: {input_format}")
        return input_format

    suffix = Path(data_path).suffix.lower()
    suffix_to_format = {
        ".xlsx": "xlsx",
        ".xls": "xlsx",
        ".csv": "csv",
        ".jsonl": "jsonl",
        ".json": "jsonl",
    }
    if suffix not in suffix_to_format:
        raise ValueError(
            f"Cannot infer format from suffix {suffix}. "
            "Please set --input-format explicitly."
        )
    return suffix_to_format[suffix]


def resolve_sheet_name(sheet_name: str | None) -> Any:
    """Resolve sheet argument to pandas-friendly value.

    Args:
        sheet_name: Raw sheet argument.

    Returns:
        Any: 0 for first sheet, integer for numeric strings, or original value.
    """
    if sheet_name is None:
        return 0
    if isinstance(sheet_name, str) and sheet_name.isdigit():
        return int(sheet_name)
    return sheet_name


def _load_row_from_jsonl(data_path: str, index: int) -> dict[str, Any]:
    """Load one row from JSONL by index.

    Args:
        data_path: JSONL path.
        index: Zero-based row index.

    Returns:
        dict[str, Any]: Parsed row.
    """
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")

    with open(data_path, "r", encoding = "utf-8") as file:
        for i, line in enumerate(file):
            if i != index:
                continue
            stripped = line.strip()
            if not stripped:
                return {}
            item = json.loads(stripped)
            if not isinstance(item, dict):
                raise ValueError(f"JSONL row at index {index} is not a JSON object")
            return item

    raise ValueError(f"Index {index} out of range for JSONL file: {data_path}")


def _load_row_from_table(
    data_path: str,
    table_format: str,
    index: int,
    sheet_name: str | None
) -> dict[str, Any]:
    """Load one row from CSV/XLSX by index.

    Args:
        data_path: Input path.
        table_format: Either "csv" or "xlsx".
        index: Zero-based row index.
        sheet_name: Sheet name/index for xlsx.

    Returns:
        dict[str, Any]: Selected row as dict.
    """
    if table_format == "csv":
        df = pd.read_csv(data_path)
    elif table_format == "xlsx":
        df = pd.read_excel(data_path, sheet_name = resolve_sheet_name(sheet_name))
    else:
        raise ValueError(f"Unsupported table_format: {table_format}")

    if index < 0 or index >= len(df):
        raise ValueError(f"Index {index} out of range. Total rows: {len(df)}")

    df.columns = [str(c).strip() for c in df.columns]
    row = df.iloc[index].to_dict()
    return {str(k).strip(): v for k, v in row.items()}


def load_single_row(
    data_path: str,
    input_format: str,
    index: int,
    sheet_name: str | None
) -> dict[str, Any]:
    """Load one row from a tabular dataset.

    Args:
        data_path: Input dataset path.
        input_format: "auto" or explicit format.
        index: Zero-based row index.
        sheet_name: Sheet option for xlsx.

    Returns:
        dict[str, Any]: Loaded row.
    """
    resolved_format = resolve_data_format(data_path = data_path, input_format = input_format)
    logger.info("resolved_input_format = %s", resolved_format)

    if resolved_format == "jsonl":
        return _load_row_from_jsonl(data_path = data_path, index = index)
    return _load_row_from_table(
        data_path = data_path,
        table_format = resolved_format,
        index = index,
        sheet_name = sheet_name,
    )


def read_text_file(file_path: str | None) -> str:
    """Read UTF-8 text file.

    Args:
        file_path: Optional file path.

    Returns:
        str: File content or empty string when path is None.
    """
    if file_path is None:
        return ""

    with open(file_path, "r", encoding = "utf-8") as file:
        return file.read().strip()


def safe_str(value: Any) -> str:
    """Convert value into printable string.

    Args:
        value: Any raw value.

    Returns:
        str: Converted string.
    """
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def build_default_user_content(row: dict[str, Any]) -> str:
    """Build default user message content from available fields.

    Args:
        row: Input sample dictionary.

    Returns:
        str: User content string.
    """
    if "article" in row and "comment" in row:
        return f"Article:\n{safe_str(row.get('article'))}\n\nComment:\n{safe_str(row.get('comment'))}"

    lines = []
    for key, value in row.items():
        lines.append(f"{key}: {safe_str(value)}")
    return "\n".join(lines)


def build_user_content(
    row: dict[str, Any],
    text_col: str | None,
    user_template: str | None
) -> str:
    """Build user message content from row.

    Args:
        row: Input sample dictionary.
        text_col: Optional column name for direct text mode.
        user_template: Optional format template with {column_name} placeholders.

    Returns:
        str: Rendered user content.
    """
    if text_col is not None:
        if text_col not in row:
            raise ValueError(f"text_col '{text_col}' not found in sample keys: {list(row.keys())}")
        return safe_str(row[text_col])

    if user_template:
        normalized = {key: safe_str(value) for key, value in row.items()}
        try:
            return user_template.format(**normalized)
        except KeyError as exc:
            missing = str(exc).strip("'")
            raise ValueError(
                f"user_template contains missing field '{missing}'. Available keys: {list(row.keys())}"
            ) from exc

    return build_default_user_content(row)


def get_label_text(row: dict[str, Any], label_col: str | None) -> str | None:
    """Read optional label text from row.

    Args:
        row: Input sample dictionary.
        label_col: Label column name.

    Returns:
        str | None: Label text if available.
    """
    if not label_col:
        return None
    if label_col not in row:
        logger.warning("label_col '%s' not found in sample keys", label_col)
        return None
    return safe_str(row[label_col]).strip() or None
