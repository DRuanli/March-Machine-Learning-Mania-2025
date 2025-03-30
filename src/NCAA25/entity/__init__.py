from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataPreparationConfig:
    data_path: Path
    teams_file: Path
    seeds_file: Path
    regular_season_results_file: Path
    tourney_results_file: Path
    detailed_results_file: Path
    output_dir: Path