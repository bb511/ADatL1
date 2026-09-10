from pathlib import Path

import pandas as pd

from src.analysis.fet_et_value_counts import ValueCountPlotter, ValueCountSpecs


def test_load_values_reads_input_variables_source_csv(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "input_variables.csv"
    pd.DataFrame({"FET.Et": [1.0, 2.0, 2.0]}).to_csv(source_path, index=False)

    plotter = ValueCountPlotter(
        ValueCountSpecs(
            variables_csv=source_path,
            column="FET.Et",
        )
    )

    pd.testing.assert_series_equal(
        plotter._load_values(),
        pd.Series([1.0, 2.0, 2.0], name="FET.Et"),
    )
