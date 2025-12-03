# Processes the data.
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
import functools
import operator

import awkward as ak
import pyarrow.parquet as parquet
from colorama import Fore, Back, Style

from src.utils import pylogger
from . import plots

log = pylogger.RankedLogger(__name__)


@dataclass
class L1DataProcessor:
    extracted_folder: str
    event_filters: dict
    object_filters: dict
    cache_root_dir: str = "data/"
    name: str = "default"
    verbose: bool = False

    def process(self, data_category: str):
        """Applies processing to the data and caches the processed data.

        :param dataset: Dictionary with keys corresponding to the name of the processed
            data set and values corresponding to the path to the raw data.
        :param data_category: String specifying the kind of data that is being extracted
            e.g., 'zerobias', 'background', or 'signal'.
        """
        log.info(Fore.GREEN + f"Processing {data_category} data...")

        self.extracted_datapath = Path(self.extracted_folder) / data_category
        self.processed_dir = Path(self.cache_root_dir) / 'processed'
        self.cache_folder = self.processed_dir / self.name / data_category
        self.existence_warn_trigger = False

        # Process the data sets.
        nexists = 0
        ndatasets = 0
        for extracted_dataset in self.extracted_datapath.iterdir():
            ndatasets += 1
            dataset_name = extracted_dataset.stem
            self.cache_dataset_folder = self.cache_folder / dataset_name
            self.event_masks_folder = self.cache_dataset_folder / 'event_masks'
            self.object_masks_folder = self.cache_dataset_folder / 'object_masks'
            if self._check_data_exists():
                nexists += 1
                continue

            self.object_names = self._get_object_names(extracted_dataset)
            self._cache_event_masks(extracted_dataset)
            self._cache(extracted_dataset)
            self._plot()

        if nexists == ndatasets:
            log.info(Fore.YELLOW + f"Processed data exists in {self.cache_folder}!")

        if self.existence_warn_trigger:
            log.warn(
                "The code only checks if there exist parquet files with the names of "
                "objects specified in the data/data_extractor config for the filters. "
                "If you are unsure the actual applied filters for each object are the "
                "same, then double check this."
            )

    def _get_object_names(self, dataset_path: Path) -> list[str]:
        """Get all the object names of the data to be processed."""
        obj_names = []
        for object_file in dataset_path.glob('*.parquet'):
            obj_names.append(object_file.stem)

        return obj_names

    def _cache_event_masks(self, extracted_dataset: Path):
        """Generate a set of masks for all the events in the data.

        Apply the filters from the event_filters dictionary to the relevant objects and
        save the corresponding mask to a designated directory. At the end, apply a
        logical AND between all the generated masks and save the result.
        """
        self.event_masks_folder.mkdir(parents=True, exist_ok=True)
        for obj_name, criterion in self.event_filters.items():
            self._check_object_exists(obj_name)
            data = ak.from_parquet(extracted_dataset / f'{obj_name}.parquet')
            context = {feature: data[feature] for feature in data.fields}
            mask = ak.numexpr.evaluate(criterion, context)
            mask = ak.all(mask, axis=1)

            ak.to_parquet(mask, self.event_masks_folder / f'{obj_name}.parquet')
        self._intersection(self.event_masks_folder)

    def _intersection(self, masks_folder: Path):
        """Applies AND logical operator between all masks in a folder, saves result."""
        masks = []
        for mask_path in masks_folder.glob("*.parquet"):
            masks.append(ak.from_parquet(mask_path))
        if not masks:
            raise ValueError(Fore.RED + f"No files found in {masks_folder}.")

        intersection = functools.reduce(operator.and_, masks)
        ak.to_parquet(intersection, self.event_masks_folder / 'intersection.parquet')

    def _cache(self, extracted_dataset: Path):
        """Processes the data and caches it.

        First, this applies the event-level mask generated previously to the data.
        Then, a mask is generated for the per-object cuts, cached to disk, and applied
        to the data as well.
        The result of this two operations is then cached to disk.
        """
        event_mask = ak.from_parquet(self.event_masks_folder / 'intersection.parquet')
        object_masks_folder.mkdir(parents=True, exist_ok=True)
        for obj_name in self.object_names:
            data = ak.from_parquet(extracted_dataset / f'{obj_name}.parquet')
            data = data[event_mask]

            context = {feature: data[feature] for feature in data.fields}
            if obj_name in self.object_filters.keys():
                criterion = self.object_filters[obj_name]
                obj_mask = ak.numexpr.evaluate(criterion, context)
                ak.to_parquet(obj_mask, object_masks_folder / f'{obj_name}.parquet')
                data = data[obj_mask]

            ak.to_parquet(data, self.cache_dataset_folder / f'{obj_name}.parquet')

        log.info("Cached proc data: " + Fore.GREEN + f"{self.cache_dataset_folder}.")

    def _check_object_exists(self, obj_name: str):
        """Check if the object to filter on exists in the extracted data."""
        if obj_name in self.object_names:
            return

        raise ValueError(Fore.RED + f"{obj_name} not in data: {self.object_names}")

    def _check_data_exists(self) -> bool:
        """Check if a specific data set was already processed.

        This also checks if the same objects were used in applying the event filters
        and the object filters as given in the config. If this is not true, the data
        is re-generated. NOTICE: this does not check if the SAME CONDITIONS were applied
        on the features of the considered objects.
        """
        if self.cache_dataset_folder.exists():
            event_masks_exists = self._check_event_masks()
            object_masks_exists = self._check_object_masks()
            if event_masks_exists and object_masks_exists:
                if self.verbose:
                    self.existence_warn_trigger = True
                    log.info(Fore.YELLOW + f"Processed data exists: {cache_path}.")

                return True

        self.cache_dataset_folder.mkdir(parents=True, exist_ok=True)
        return False

    def _check_event_masks(self) -> bool:
        """Check whether all the event masks given in the config already exist."""
        if not self.event_masks_folder.exists():
            return False

        for obj_name in self.event_filters.keys():
            event_mask_file = self.event_masks_folder / f'{obj_name}.parquet'
            if not event_mask_file.is_file():
                return False

        return True

    def _check_object_masks(self) -> bool:
        """Check whether all the object masks given in the config already exist."""
        if not self.object_masks_folder.exists():
            return False

        for obj_name in self.object_filters.keys():
            object_mask_file = self.object_masks_folder / f'{obj_name}.parquet'
            if not object_mask_file.is_file():
                return False

        return True

    def _plot(self):
        """Read and then plot the processed data."""
        plots_dir = self.cache_dataset_folder / 'PLOTS'

        for object_file in self.cache_dataset_folder.glob('*.parquet'):
            data = ak.from_parquet(object_file)
            obj_folder = plots_dir / object_file.stem
            obj_folder.mkdir(parents=True, exist_ok=True)

            for feature in data.fields:
                plots.plot_hist(data[feature], feature, obj_folder)
