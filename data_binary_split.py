#!/usr/bin/env python3
"""Generic binary dataset split utility.

Supported input formats: xlsx, csv.
Supported split modes:
- ratio: split each class by train ratio
- count: split by explicit train/test class counts
"""

import argparse
import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


DEFAULT_TRUE_LABELS = "是,true,1,yes,y,t"
DEFAULT_FALSE_LABELS = "否,false,0,no,n,f"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description = "Split binary dataset into train/test with class balance control"
    )
    parser.add_argument(
        "--input-path",
        required = True,
        help = "Input data file path (xlsx/csv)"
    )
    parser.add_argument(
        "--input-format",
        default = "auto",
        choices = ["auto", "xlsx", "csv"],
        help = "Input format"
    )
    parser.add_argument(
        "--sheet-name",
        default = None,
        help = "Sheet name/index for xlsx input"
    )
    parser.add_argument(
        "--label-col",
        required = True,
        help = "Label column name"
    )
    parser.add_argument(
        "--output-train-path",
        default = None,
        help = "Train output path. Default: <input_stem>-train.<format>"
    )
    parser.add_argument(
        "--output-test-path",
        default = None,
        help = "Test output path. Default: <input_stem>-test.<format>"
    )
    parser.add_argument(
        "--output-format",
        default = "auto",
        choices = ["auto", "xlsx", "csv"],
        help = "Output format. auto follows output suffix or input format"
    )
    parser.add_argument(
        "--split-mode",
        default = "ratio",
        choices = ["ratio", "count"],
        help = "Split mode"
    )
    parser.add_argument(
        "--train-ratio",
        type = float,
        default = 0.8,
        help = "Train ratio for ratio mode"
    )
    parser.add_argument("--train-pos", type = int, default = 0)
    parser.add_argument("--train-neg", type = int, default = 0)
    parser.add_argument("--test-pos", type = int, default = 0)
    parser.add_argument("--test-neg", type = int, default = 0)
    parser.add_argument(
        "--positive-labels",
        default = DEFAULT_TRUE_LABELS,
        help = "Comma-separated label values treated as positive"
    )
    parser.add_argument(
        "--negative-labels",
        default = DEFAULT_FALSE_LABELS,
        help = "Comma-separated label values treated as negative"
    )
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument(
        "--astype-str-cols",
        default = "",
        help = "Comma-separated columns to convert to string before saving"
    )
    return parser.parse_args()


def parse_label_set(values: str) -> set[str]:
    """Parse comma-separated labels into a normalized set.

    Args:
        values: Comma-separated labels.

    Returns:
        set[str]: Normalized lower-case labels.
    """
    return {
        item.strip().lower()
        for item in str(values).split(",")
        if item.strip()
    }


def normalize_label(value, true_set: set[str], false_set: set[str]) -> bool | None:
    """Normalize label value to binary bool.

    Args:
        value: Raw label value.
        true_set: Positive label set.
        false_set: Negative label set.

    Returns:
        bool | None: Normalized label or None when unknown.
    """
    if value is None or pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        numeric = str(int(value))
        if numeric in true_set:
            return True
        if numeric in false_set:
            return False

    text = str(value).strip().lower()
    if text in true_set:
        return True
    if text in false_set:
        return False
    return None


def resolve_sheet_name(sheet_name: str | None):
    """Resolve sheet option.

    Args:
        sheet_name: Raw sheet option.

    Returns:
        Any: Pandas-compatible sheet selector.
    """
    if sheet_name is None:
        return 0
    if isinstance(sheet_name, str) and sheet_name.isdigit():
        return int(sheet_name)
    return sheet_name


def resolve_file_format(path: str, provided_format: str, fallback: str | None = None) -> str:
    """Resolve data format from argument/suffix.

    Args:
        path: File path.
        provided_format: Explicit or auto.
        fallback: Fallback format if suffix is absent.

    Returns:
        str: Resolved format.
    """
    if provided_format != "auto":
        return provided_format

    suffix = Path(path).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return "xlsx"
    if suffix == ".csv":
        return "csv"

    if fallback is not None:
        return fallback
    raise ValueError(f"Cannot infer format from path: {path}")


def read_dataframe(input_path: str, input_format: str, sheet_name: str | None) -> pd.DataFrame:
    """Read input dataframe.

    Args:
        input_path: Input path.
        input_format: auto/xlsx/csv.
        sheet_name: Optional xlsx sheet selector.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    resolved_format = resolve_file_format(path = input_path, provided_format = input_format)

    if resolved_format == "xlsx":
        df = pd.read_excel(input_path, sheet_name = resolve_sheet_name(sheet_name))
    elif resolved_format == "csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input format: {resolved_format}")

    df.columns = [str(col).strip() for col in df.columns]
    return df


def apply_astype_str(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert selected columns to string.

    Args:
        df: Input dataframe.
        cols: Target columns.

    Returns:
        pd.DataFrame: Converted dataframe.
    """
    if not cols:
        return df

    output = df.copy()
    for col in cols:
        if col in output.columns:
            output[col] = output[col].astype(str)
    return output


def split_by_ratio(
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    train_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split pos/neg samples by ratio.

    Args:
        pos_df: Positive samples.
        neg_df: Negative samples.
        train_ratio: Train ratio in (0, 1).
        seed: Random seed.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train/Test dataframes.
    """
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    def _train_count(total: int) -> int:
        if total < 2:
            raise ValueError("Each class must contain at least 2 rows in ratio mode")
        count = int(total * train_ratio)
        if count <= 0:
            return 1
        if count >= total:
            return total - 1
        return count

    train_pos = _train_count(len(pos_df))
    train_neg = _train_count(len(neg_df))

    train_pos_df = pos_df.sample(n = train_pos, random_state = seed)
    train_neg_df = neg_df.sample(n = train_neg, random_state = seed + 1)

    test_pos_df = pos_df.drop(index = train_pos_df.index)
    test_neg_df = neg_df.drop(index = train_neg_df.index)

    train_df = pd.concat([train_pos_df, train_neg_df], axis = 0)
    test_df = pd.concat([test_pos_df, test_neg_df], axis = 0)

    train_df = train_df.sample(frac = 1.0, random_state = seed + 2).reset_index(drop = True)
    test_df = test_df.sample(frac = 1.0, random_state = seed + 3).reset_index(drop = True)
    return train_df, test_df


def split_by_count(
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    train_pos: int,
    train_neg: int,
    test_pos: int,
    test_neg: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split pos/neg samples by explicit counts.

    Args:
        pos_df: Positive samples.
        neg_df: Negative samples.
        train_pos: Train positive count.
        train_neg: Train negative count.
        test_pos: Test positive count.
        test_neg: Test negative count.
        seed: Random seed.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train/Test dataframes.
    """
    need_pos = train_pos + test_pos
    need_neg = train_neg + test_neg
    if len(pos_df) < need_pos:
        raise ValueError(f"Not enough positives: {len(pos_df)} < {need_pos}")
    if len(neg_df) < need_neg:
        raise ValueError(f"Not enough negatives: {len(neg_df)} < {need_neg}")

    test_pos_df = pos_df.sample(n = test_pos, random_state = seed)
    remain_pos_df = pos_df.drop(index = test_pos_df.index)
    train_pos_df = remain_pos_df.sample(n = train_pos, random_state = seed + 1)

    test_neg_df = neg_df.sample(n = test_neg, random_state = seed + 2)
    remain_neg_df = neg_df.drop(index = test_neg_df.index)
    train_neg_df = remain_neg_df.sample(n = train_neg, random_state = seed + 3)

    train_df = pd.concat([train_pos_df, train_neg_df], axis = 0)
    test_df = pd.concat([test_pos_df, test_neg_df], axis = 0)

    train_df = train_df.sample(frac = 1.0, random_state = seed + 4).reset_index(drop = True)
    test_df = test_df.sample(frac = 1.0, random_state = seed + 5).reset_index(drop = True)
    return train_df, test_df


def resolve_output_paths(
    input_path: str,
    output_train_path: str | None,
    output_test_path: str | None,
    output_format: str,
    input_format: str,
) -> tuple[str, str, str]:
    """Resolve train/test output paths and final output format.

    Args:
        input_path: Input file path.
        output_train_path: Optional train output path.
        output_test_path: Optional test output path.
        output_format: Requested output format.
        input_format: Resolved input format.

    Returns:
        tuple[str, str, str]: (train_path, test_path, resolved_output_format)
    """
    input_obj = Path(input_path)

    if output_train_path is None or output_test_path is None:
        inferred_ext = ".xlsx" if input_format == "xlsx" else ".csv"
        if output_train_path is None:
            output_train_path = str(input_obj.with_name(input_obj.stem + "-train" + inferred_ext))
        if output_test_path is None:
            output_test_path = str(input_obj.with_name(input_obj.stem + "-test" + inferred_ext))

    if output_format == "auto":
        resolved_output_format = resolve_file_format(
            path = output_train_path,
            provided_format = "auto",
            fallback = input_format,
        )
    else:
        resolved_output_format = output_format

    return output_train_path, output_test_path, resolved_output_format


def save_dataframe(df: pd.DataFrame, output_path: str, output_format: str) -> None:
    """Save dataframe in requested format.

    Args:
        df: Dataframe to save.
        output_path: Output file path.
        output_format: xlsx/csv.
    """
    output_obj = Path(output_path)
    output_obj.parent.mkdir(parents = True, exist_ok = True)

    if output_format == "xlsx":
        with pd.ExcelWriter(output_obj, engine = "openpyxl") as writer:
            df.to_excel(writer, index = False)
        return

    if output_format == "csv":
        df.to_csv(output_obj, index = False)
        return

    raise ValueError(f"Unsupported output format: {output_format}")


def main() -> None:
    """CLI entry."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Binary Data Split")
    logger.info("=" * 80)

    input_format = resolve_file_format(
        path = args.input_path,
        provided_format = args.input_format,
    )
    df = read_dataframe(
        input_path = args.input_path,
        input_format = input_format,
        sheet_name = args.sheet_name,
    )

    label_col = args.label_col.strip()
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}. Available: {list(df.columns)}")

    true_set = parse_label_set(args.positive_labels)
    false_set = parse_label_set(args.negative_labels)
    overlap = true_set.intersection(false_set)
    if overlap:
        raise ValueError(f"positive_labels and negative_labels overlap: {sorted(overlap)}")

    normalized = df[label_col].apply(
        lambda value: normalize_label(value, true_set = true_set, false_set = false_set)
    )
    if normalized.isna().any():
        unknown_values = df.loc[normalized.isna(), label_col].astype(str).value_counts(dropna = False).to_dict()
        raise ValueError(f"Unrecognized label values in {label_col}: {unknown_values}")

    pos_df = df[normalized == True]
    neg_df = df[normalized == False]
    logger.info("positive_count = %d, negative_count = %d", len(pos_df), len(neg_df))

    if args.split_mode == "ratio":
        train_df, test_df = split_by_ratio(
            pos_df = pos_df,
            neg_df = neg_df,
            train_ratio = args.train_ratio,
            seed = args.seed,
        )
    else:
        train_df, test_df = split_by_count(
            pos_df = pos_df,
            neg_df = neg_df,
            train_pos = args.train_pos,
            train_neg = args.train_neg,
            test_pos = args.test_pos,
            test_neg = args.test_neg,
            seed = args.seed,
        )

    cast_cols = [col.strip() for col in str(args.astype_str_cols).split(",") if col.strip()]
    train_df = apply_astype_str(df = train_df, cols = cast_cols)
    test_df = apply_astype_str(df = test_df, cols = cast_cols)

    train_path, test_path, resolved_output_format = resolve_output_paths(
        input_path = args.input_path,
        output_train_path = args.output_train_path,
        output_test_path = args.output_test_path,
        output_format = args.output_format,
        input_format = input_format,
    )

    save_dataframe(df = train_df, output_path = train_path, output_format = resolved_output_format)
    save_dataframe(df = test_df, output_path = test_path, output_format = resolved_output_format)

    logger.info("train_path = %s, train_rows = %d", train_path, len(train_df))
    logger.info("test_path = %s, test_rows = %d", test_path, len(test_df))


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
