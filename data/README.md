# Data

Everything the experiments need ships with this repository. The large raw
satellite archive is fetched separately from Zenodo.

```
data/
├── download_data.py            fetches raw/ from Zenodo
├── raw/
│   ├── all_training.geojson    1,000 labeled lake outlines        [tracked]
│   ├── all_lakes_2019.nc       S1/S2 daily time series, 280 MB     [download]
│   └── all_lakes_2018.nc       optional, unused by the paper       [download]
└── processed/
    ├── lake_manifest.csv                    provenance of every labeled lake
    ├── backscatter_777.csv                  evaluation set, S1 only
    ├── backscatter_water_777.csv            evaluation set, S1 + S2
    ├── backscatter_representative.csv       training set, S1 only
    └── backscatter_water_representative.csv training set, S1 + S2
```

## Getting the raw data

```bash
python data/download_data.py
```

Downloads about 65 MB from Zenodo, verifies MD5 checksums, and extracts
`all_lakes_2019.nc` and `all_training.geojson` into `data/raw/`. Only
`notebooks/01_preprocessing` needs these; every experiment in the paper runs
from `data/processed/` alone.

## Rebuilding the processed CSVs

```bash
python scripts/build_processed_data.py --check    # verify only
python scripts/build_processed_data.py --write    # regenerate
```

The pipeline is:

| Channel | Steps |
| --- | --- |
| $HV_{anom}$ | centered 12-day rolling mean on `HV_lake` and `HV_out` → linear interpolation between observations → difference → zero-fill any remainder |
| $p_{water}$ | linear interpolation of `S2_water` between observations only → zero-fill leading and trailing gaps |

`--check` confirms this reproduces the shipped CSVs for all 777 evaluation
lakes to within floating-point round-off. The smoothing filter is the one
described in Section 4.1 of the paper; note that it runs *before* interpolation,
so interpolated values never feed back into the filter.

## Provenance

| Layer | Source | License |
| --- | --- | --- |
| Lake outlines | [Dunmire et al. 2021](https://doi.org/10.5281/zenodo.4813833) | CC BY 4.0 |
| S1/S2 time series (`all_lakes_2019.nc`) | [Dunmire et al. 2025](https://doi.org/10.5281/zenodo.14587026) | CC BY 4.0 |
| Manual class labels, lake selection | This work | CC BY 4.0 |

`all_lakes_2019.nc` as distributed here is byte-for-byte identical, over every
variable this pipeline reads (`HV_lake`, `HV_out`, `S2_water`), to the file used
to produce the published results. See [DATA_LICENSE.md](../DATA_LICENSE.md).

## Selection chain

`lake_manifest.csv` records how the 1,000 labeled lakes became the 777 used in
the paper.

| Stage | Lakes | Column | How |
| --- | --- | --- | --- |
| Labeled | 1,000 | — | manual labeling of all six GrIS subregions, 250 per class |
| Visual QC | 793 | `in_visual_qc` | manual review of each lake's plotted time series |
| Final | 777 | `in_final_777` | evaluation set reported in the paper |

The 16 lakes dropped between visual QC and the final set were a manual
curation decision with no recoverable programmatic rule, so the final set is
distributed as an explicit manifest rather than as a filter to re-run. The
resulting class balance matches the paper exactly: 189 refreeze, 392 drain,
196 buried.

`label_4class` keeps the original `drain` / `drain_slow` distinction;
`label_3class` merges them into `drain` for the three-class problem studied in
the paper.

## File schema

**`*_777.csv` and `*_representative.csv`** — one lake per row.

| Column | Description |
| --- | --- |
| `ids` | Lake identifier, `<REGION><YEAR>_<n>`, e.g. `CW2019_1967` |
| `backscatter_diff_1` … `_365` | Daily $HV_{anom} = HV_{lake} - HV_{background}$, dB |
| `water_percentage_1` … `_365` | Daily $p_{water}$, percent of lake pixels classed as water (combined files only) |
| `label` | `refreeze`, `drain`, or `buried` |

Days are day-of-year. Values are linearly interpolated between satellite
observations, and $HV_{anom}$ carries a 12-day smoothing filter (paper,
Section 4.1). The experiments use days 121–364, the melt-season window.

> **Row order.** `backscatter_777.csv` and `backscatter_water_777.csv` hold the
> same 777 lakes with the same labels and identical backscatter values, but the
> rows are in different orders. Each file is internally consistent, so this
> affects nothing as long as you read `ids`, the series, and `label` from the
> same file — which `rpsgmm.data` does. Join on `ids` if you need to combine
> them.

**`lake_manifest.csv`** — `ids`, `region`, `year`, `area_m2`, `elevation_m`,
`label_4class`, `label_3class`, `in_visual_qc`, `in_final_777`.

## Citation

Please cite the paper and the Zenodo record together:

```bibtex
@dataset{dunmire2025gris,
  author    = {Dunmire, Devon and Subramanian, Aneesh and Hossain, Emam and
               Gani, Md Osman and Banwell, Alison and Younas, Hammad and Myers, Brendan},
  title     = {{Data and Code for: Greenland Ice Sheet wide supraglacial lake
               evolution and dynamics: insights from the 2018 and 2019 melt seasons}},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.14587026}
}
```
