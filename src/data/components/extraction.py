# Extracts the data from the raw parquet files.
from pathlib import Path
from dataclasses import dataclass

import awkward as ak
import pyarrow.parquet as parquet
from omegaconf import OmegaConf
from colorama import Fore

from adl1t_datamaker.convert.loader import Parquet2Awkward
from src.utils import pylogger
from . import plots

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@dataclass
class L1DataExtractor(object):
    """Extract data from level 1 global trigger data files saved in parquet format.

    This class extracts the data using the Parquet2Awkward reader from the
    adl1t_datamaker package. This reader reads folders of parquet files generated with
    the adl1t_datamaker package. The folder structure is fixed.

    Each subfolder within a dataset folder corresponds to an object, e.g.,
    muons, egammas, etc., each with multiple parquet files. This class merges all the
    parquet files corresponding to an object into one parquet file corresponding to that
    object for easier reading down the line. This merging is done sequentially and does
    not bloat the memory.

    :param select_features: Dictionary with which features to extract from the
        parquet file (keys), and the corresponding features to get from the object.
        If a key corresponds to 'none' or is missing from this dictionary but is
        present in the data, then do not load that object.
    :param feat_name_map: Dictionary that maps original field names for each object in
        the parquet files to new ones for easier readability and access.
    :param cache_root_dir: String specifying path to where to store the data.
    :param name: String used to name a specific method of doing the extraction of
        the data, e.g., "default", "axov4".
    """

    select_features: dict
    feat_name_map: dict
    cache_root_dir: str = "data"
    name: str = "default"
    verbose: bool = False

    def extract(self, datasets: dict, data_category: str):
        """Extract the data.

        Args:
            datasets: Dictionary containing the name of the datasets and the
                corresponding folder path associated with it.
            data_category: String specifying the kind of data that is being extracted
                e.g., 'zerobias', 'background', or 'signal'.
        """
        log.info(Fore.GREEN + f"Extracting {data_category} data.")
        self.extracted_dir = Path(self.cache_root_dir) / "extracted"
        self.cache_folder = self.extracted_dir / self.name / data_category
        self.select_feats = OmegaConf.to_container(self.select_features, resolve=True)
        self.existence_warn_trigger = False

        nexists = 0
        for dataset_name, folder_path in datasets.items():
            existing_objs = self._check_data_exists(dataset_name)
            if len(existing_objs) == len(self.select_feats.keys()):
                nexists += 1
                continue

            extr_objs = {
                obj_name: obj_feats
                for obj_name, obj_feats in self.select_feats.items()
                if obj_name not in existing_objs
            }

            data = Parquet2Awkward(folder_path, select_feats=extr_objs, threading=False)
            self._cache(data, dataset_name)

        if nexists == len(datasets.keys()):
            log.info(Fore.YELLOW + f"Extracted data exists in {self.cache_folder}!")

        if self.existence_warn_trigger:
            log.warn(
                "The code only checks if there exist parquet files with the names of "
                "objects specified in the data/data_extractor config. If you are "
                "unsure whether the features are as expected, double check this."
            )

    def _cache(self, data: Parquet2Awkward, dataset_name: str):
        """Store the extracted data in a folder corresponding to the dataset name."""
        dataset_folder = self.cache_folder / dataset_name
        dataset_folder.mkdir(parents=True, exist_ok=True)

        for obj_name in data.object_names:
            cache_file = dataset_folder / f"{obj_name}.parquet"
            parquet_writer = self._write_parquet(data, cache_file, obj_name)
            parquet_writer.close()
            self._plot(dataset_name, obj_name)

        log.info("Cached extracted data at " + Fore.GREEN + f"{dataset_folder}.")

    def _write_parquet(self, data, cache_file: Path, obj_name: str, writer=None):
        """Stream batches of given obj in data to a single parquet file."""
        for batch in data(obj_name):
            batch = self._rename_features(batch, obj_name)
            batch = ak.to_arrow_table(batch)
            if writer is None:
                writer = parquet.ParquetWriter(
                    cache_file, batch.schema, compression="snappy"
                )
            writer.write_table(batch)

        return writer

    def _check_data_exists(self, dataset_name: str) -> set[str]:
        """Check if a specific data set was already extracted."""
        dataset_folder = self.cache_folder / dataset_name

        if dataset_folder.is_dir():
            dataset_files = [
                p for p in dataset_folder.iterdir() if not p.name.startswith("._")
            ]
            if any(dataset_files):
                existing_objs = self._get_existing_objs(dataset_folder)
                if self.verbose:
                    self.existence_warn_trigger = True
                    log.info(
                        Fore.YELLOW + f"Extracted data exists: {dataset_folder}. "
                        f"Existing objects {existing_objs}."
                    )
                return existing_objs

        return set()

    def _get_existing_objs(self, dataset_folder: Path) -> set[str]:
        """Checks if all objects specified in the config have been extracted."""
        existing_objs = set()
        for obj_name in self.select_features.keys():
            obj_cache_filepath = dataset_folder / f"{obj_name}.parquet"
            if obj_cache_filepath.is_file():
                existing_objs.add(obj_name)

        return existing_objs

    def _rename_features(self, batch: ak.Array, obj_name: str):
        """Rename the data features for object."""
        if obj_name in self.feat_name_map.keys():
            mapping = self.feat_name_map[obj_name]
            batch = ak.zip(
                {mapping.get(feat, feat): batch[feat] for feat in batch.fields}
            )

        return batch

    def _plot(self, dataset_name: str, obj_name: str):
        """Read and then plot the extracted data."""
        data_dir = self.cache_folder / dataset_name
        plots_dir = self.cache_folder / dataset_name / "PLOTS"
        object_file = data_dir / f"{obj_name}.parquet"

        data = ak.from_parquet(object_file)
        obj_folder = plots_dir / obj_name
        obj_folder.mkdir(parents=True, exist_ok=True)

        for feature in data.fields:
            plots.plot_hist(data[feature], feature, obj_folder)
