import pathlib

import pandas as pd

_filepath = pathlib.Path(__file__).parents[0] / "data/periodic_table.parquet"


def get_periodic_table(masses_only: bool = True):
    """
    Docstring
    """

    if not masses_only:
        return pd.read_parquet(_filepath)

    return pd.read_parquet(_filepath, columns=["AtomicMass"]).squeeze()
