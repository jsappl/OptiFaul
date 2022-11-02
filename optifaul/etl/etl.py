"""Extract, transform, load."""

from pathlib import Path

from . import extract as E
from . import load as L
from . import transform as T


class ETL:
    """The Extract, Transform, Load class."""

    def __init__(self, data_path: str) -> None:
        """Initialize ETL pipeline parameters.

        Args:
            data_path: Path to directory containing the raw data.
        """
        self.data_path = Path(data_path)

    def _extract(self) -> None:
        """Load raw data from source locations to the staging area."""
        self.dfs = E.load_df(self.data_path)

    def _transform(self) -> None:
        """Transform raw data in multiple steps."""
        self.dfs = T.keep_translate_cols(self.dfs)
        self.dfs = T.string_to_datetime(self.dfs)
        self.df = T.merge(self.dfs)
        self.df = T.sort_by_date(self.df)
        self.df = T.interpolate_nans(self.df)
        self.df = T.smoothen(self.df, ndays=7)
        # self.df = T.treat_outliers(self.df, method="cap")
        self.df = T.flatten_digesters(self.df)
        self.df = T.enrich(self.df)

    def _load(self) -> None:
        """Move transformed data from the staging area to a target location."""
        L.to_disk(self.df)

    def execute(self) -> None:
        """Run all ETL steps."""
        self._extract()
        self._transform()
        self._load()
