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
        for extracted_dataset in sorted(
            p for p in self.extracted_datapath.iterdir()
            if p.is_dir() and not p.name.startswith("._")
        ):
            ndatasets += 1
            dataset_name = extracted_dataset.stem
            self.dataset_folder = self.cache_folder / dataset_name
            self.event_masks_folder = self.dataset_folder / 'event_masks'
            self.object_masks_folder = self.dataset_folder / 'object_masks'

            extr_object_names = self._get_extracted_object_names(extracted_dataset)
            existing_objs = self._check_data_exists()
            self.object_names = extr_object_names - existing_objs

            if len(self.object_names) == 0:
                nexists += 1
                continue

            self._cache_event_masks(extracted_dataset)
            self._cache(extracted_dataset)

        if nexists == ndatasets:
            log.info(Fore.YELLOW + f"Processed data exists in {self.cache_folder}!")
        elif nexists > 0:
            log.info(Fore.YELLOW + f"Data partially exists in {self.cache_folder}!")

        if self.existence_warn_trigger:
            log.warn(
                "The code only checks if there exist parquet files with the names of "
                "objects specified in the data/data_extractor config for the filters. "
                "If you are unsure the actual applied filters for each object are the "
                "same, then double check this."
            )

    def _get_extracted_object_names(self, dataset_path: Path) -> list[str]:
        """Get all the object names of the objects that were extracted."""
        obj_names = set()
        for object_file in sorted(
            p for p in dataset_path.glob("*.parquet")
            if p.is_file() and not p.name.startswith("._")
        ):
            obj_names.add(object_file.stem)

        if not set(self.event_filters.keys()) <= obj_names:
            diff = set(self.event_filters.keys()) - obj_names
            raise FileNotFoundError(
                f"Event filter objects {diff} not in the extracted data! "
            )

        if not set(self.object_filters.keys()) <= obj_names:
            diff = set(self.object_filters.keys()) - obj_names
            raise FileNotFoundError(
                f"Object filter objects {diff} not in the extracted data! "
            )

        return obj_names

    def _cache_event_masks(self, extracted_dataset: Path):
        """Generate a set of masks for all the events in the data.

        Apply the filters from the event_filters dictionary to the relevant objects and
        save the corresponding mask to a designated directory. At the end, apply a
        logical AND between all the generated masks and save the result.
        """
        self.event_masks_folder.mkdir(parents=True, exist_ok=True)
        for obj_name, criterion in self.event_filters.items():
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
            if not mask_path.is_file() or mask_path.name.startswith("._"):
                continue
            
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
        self.object_masks_folder.mkdir(parents=True, exist_ok=True)
        for obj_name in self.object_names:
            data = ak.from_parquet(extracted_dataset / f'{obj_name}.parquet')
            data = data[event_mask]

            context = {feature: data[feature] for feature in data.fields}
            if obj_name in self.object_filters.keys():
                obj_mask = self._get_obj_mask(obj_name, context)
                data = data[obj_mask]

            ak.to_parquet(data, self.dataset_folder / f'{obj_name}.parquet')
            self._plot(data, obj_name)

        log.info("Cached proc data: " + Fore.GREEN + f"{self.dataset_folder}.")

    def _get_obj_mask(self, obj_name: str, context: dict):
        """Import the object mask or construct it if it does not exist."""
        obj_mask_file = self.object_masks_folder / f'{obj_name}.parquet'
        if obj_mask_file.is_file():
            return ak.from_parquet(obj_mask_file)

        criterion = self.object_filters[obj_name]
        obj_mask = ak.numexpr.evaluate(criterion, context)
        ak.to_parquet(obj_mask, obj_mask_file)

        return obj_mask

    def _check_data_exists(self) -> set[str]:
        """Check if a specific data set was already processed.

        Also check if the same event filters and object masks were applied in their
        processing. Return a list of existing objects, which have already been processed
        and have a mask corresponding mask file, if they're supposed to be masked.
        If the event filters differ from the ones provided in the config, redo the
        processing for all the objects.
        """
        dataset_files = [
            p for p in self.dataset_folder.iterdir()
            if not p.name.startswith("._")
        ]
        if self.dataset_folder.is_dir() and any(dataset_files):
            existing_objs = self._get_existing_objs()
            if self.verbose:
                self.existence_warn_trigger = True
                log.info(Fore.YELLOW + f"Processed data exists {existing_objs}.")

            if not self._check_event_masks():
                return set()

            missing_obj_masks = self._check_object_masks()
            existing_objs = existing_objs - missing_obj_masks

            return existing_objs

        self.dataset_folder.mkdir(parents=True, exist_ok=True)
        return set()

    def _check_event_masks(self) -> bool:
        """Check whether all the event masks given in the config already exist."""
        if not self.event_masks_folder.exists():
            return False

        for obj_name in self.event_filters.keys():
            event_mask_file = self.event_masks_folder / f'{obj_name}.parquet'
            if not event_mask_file.is_file():
                return False

        return True

    def _check_object_masks(self) -> set[str]:
        """Check whether all the object masks given in the config already exist."""
        if not self.object_masks_folder.exists():
            return set()

        missing_obj_masks = set()
        for obj_name in self.object_filters.keys():
            object_mask_file = self.object_masks_folder / f'{obj_name}.parquet'
            if not object_mask_file.is_file():
                missing_obj_masks.add(obj_name)

        return missing_obj_masks

    def _get_existing_objs(self) -> set[str]:
        """Checks if all objects specified in the config have been extracted."""
        existing_objs = set()
        for obj_cache_filepath in sorted(
            p for p in self.dataset_folder.glob("*.parquet")
            if p.is_file() and not p.name.startswith("._")
        ):
            if obj_cache_filepath.is_file():
                existing_objs.add(obj_cache_filepath.stem)

        return existing_objs

    def _plot(self, data: ak.Array, obj_name: str):
        """Read and then plot the processed data."""
        plots_dir = self.dataset_folder / 'PLOTS'
        object_file = self.dataset_folder / f'{obj_name}.parquet'
        obj_folder = plots_dir / obj_name
        obj_folder.mkdir(parents=True, exist_ok=True)

        for feature in data.fields:
            plots.plot_hist(data[feature], feature, obj_folder)
