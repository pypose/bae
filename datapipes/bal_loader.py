"""
This file contains utilities for the Bundle Adjustment in the Large dataset.

The dataset is from the following paper:
Sameer Agarwal, Noah Snavely, Steven M. Seitz, and Richard Szeliski.
Bundle adjustment in the large.
In European Conference on Computer Vision (ECCV), 2010.

Link to the dataset: https://grail.cs.washington.edu/projects/bal/
"""

import bz2
import os
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

from .bal_io import DTYPE, read_bal_data

__all__ = ['get_problem', 'read_bal_data', 'DATA_URL', 'ALL_DATASETS']

# base url for the BAL dataset, used to download the problem files
DATA_URL = 'https://grail.cs.washington.edu/projects/bal/'

# all dataset names in the BAL dataset, used to check if the dataset name is valid
ALL_DATASETS = ['ladybug', 'trafalgar', 'dubrovnik', 'venice', 'final']


def _validate_dataset(dataset: str) -> None:
    if dataset not in ALL_DATASETS:
        raise ValueError(f"dataset_name must be one of {ALL_DATASETS}")


def _normalize_problem_name(problem_name: str) -> str:
    normalized = os.path.basename(problem_name)
    if normalized.endswith('.bz2'):
        normalized = normalized[:-4]
    if normalized.endswith('.txt'):
        normalized = normalized[:-4]
    return normalized


def _problem_url(dataset: str, problem_name: str) -> str:
    filename = _normalize_problem_name(problem_name) + '.txt.bz2'
    return urljoin(DATA_URL, f'data/{dataset}/{filename}')


def _problem_paths(cache_dir: Path, problem_name: str) -> tuple[Path, Path]:
    normalized = _normalize_problem_name(problem_name)
    return cache_dir / f'{normalized}.txt', cache_dir / f'{normalized}.txt.bz2'


def _download_url(url: str, destination: Path, timeout_s: float = 60.0) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + '.tmp')
    request = urllib.request.Request(url, headers={'User-Agent': 'bae-bal-loader/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response, tmp_path.open('wb') as f:
            shutil.copyfileobj(response, f)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, destination)
    return destination


def _ensure_problem_available(dataset: str, problem_name: str, cache_dir: Path) -> Path:
    txt_path, bz2_path = _problem_paths(cache_dir, problem_name)
    if txt_path.exists() and txt_path.stat().st_size > 0:
        return txt_path

    if not bz2_path.exists() or bz2_path.stat().st_size == 0:
        _download_url(_problem_url(dataset, problem_name), bz2_path)

    tmp_path = txt_path.with_suffix(txt_path.suffix + '.tmp')
    try:
        with bz2.open(bz2_path, 'rb') as src, tmp_path.open('wb') as dst:
            shutil.copyfileobj(src, dst)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, txt_path)
    return txt_path

def get_problem(problem_name, dataset, cache_dir='bal_data', use_quat=True, *, dtype=DTYPE):
    cache_path = Path(cache_dir)
    print(f"Preparing data for {dataset}...")
    _validate_dataset(dataset)
    problem_path = _ensure_problem_available(dataset, problem_name, cache_path)
    return read_bal_data(str(problem_path), use_quat=use_quat, dtype=dtype)
