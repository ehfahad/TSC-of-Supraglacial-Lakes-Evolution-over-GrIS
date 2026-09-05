# Data license

The **code** in this repository is licensed under the GNU General Public
License v3.0; see [LICENSE](LICENSE).

The **data** is licensed separately, as described below.

---

## 1. Files distributed in this repository

`data/raw/all_training.geojson` and everything under `data/processed/` are
released under the **Creative Commons Attribution 4.0 International
(CC BY 4.0)** license: <https://creativecommons.org/licenses/by/4.0/>

You are free to share and adapt this material for any purpose, including
commercially, provided you give appropriate credit, link to the license, and
indicate if changes were made.

These files are derived from the dataset published as:

> Dunmire, D., Subramanian, A., Hossain, E., Gani, M. O., Banwell, A.,
> Younas, H., & Myers, B. (2025). *Data and Code for: "Greenland Ice Sheet wide
> supraglacial lake evolution and dynamics: insights from the 2018 and 2019 melt
> seasons"* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14587026
> Licensed under CC BY 4.0.

The manual class labels (`refreeze` / `drain` / `drain_slow` / `buried`) and the
lake selection recorded in `data/processed/lake_manifest.csv` are contributed by
the authors of the ICMLA 2024 paper this repository accompanies.

## 2. Files fetched by `data/download_data.py`

`data/download_data.py` retrieves `all_lakes_2019.nc` and `all_training.geojson`
from Zenodo record [10.5281/zenodo.14587026](https://doi.org/10.5281/zenodo.14587026).
Those files are **not** redistributed here — `all_lakes_2019.nc` is 280 MB,
above GitHub's file size limit — and remain under the CC BY 4.0 license of that
record.

## 3. Upstream lake outlines

The supraglacial lake outlines underlying the dataset were first published as:

> Dunmire, D., Banwell, A. F., Wever, N., Lenaerts, J. T. M., & Datta, R. T.
> (2021). *Contrasting regional variability of buried meltwater extent over two
> years across the Greenland Ice Sheet - data* [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.4813833
> Licensed under CC BY 4.0.

## 4. Attribution

If you use this data, please cite **both** the ICMLA 2024 paper (see
[CITATION.cff](CITATION.cff)) and Zenodo record 10.5281/zenodo.14587026.
