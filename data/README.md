# Data folder

In this folder, the `LDI_ABG.csv`, `LDI_CLF.csv`, `LDI_HON.csv`, `LDI_MMB.csv`, `LDI_SFS.csv`, and `LDI_TUC.csv` files contain the Local Disturbance index (LDi) data for the observatories ABG, CLF, HON, MMB, SFS, and TUC, respectively. In those `.csv` files, the datetime column is in UTC and the MLT was calculated using the [``accgmv2``](https://pypi.org/project/aacgmv2/) python library.

The `all_timeline` subfolder contains the solar wind data from 1998 to 2024, which was obtained from the [OMNIWeb](https://omniweb.gsfc.nasa.gov/) database. The data is organized in yearly files, and the datetime column is in UTC.

## LDi calculation subfolder

The subfolder `ldi_calculation` contains LDi data files organized by observatory, year, and month.

The station folders are:

```text
abg/
clf/
hon/
mmb/
sfs/
tuc/
```

The original files were obtained in the IAGA2002 INTERMAGNET exchange format. INTERMAGNET data formats, file naming conventions, and data-quality categories are described in the [INTERMAGNET Technical Reference Manual](https://tech-man.intermagnet.org/stable/appendices/dataformats.html).

### Columns

Each corrected CSV file contains the following columns:

| Column        | Description                                                                                                                                                                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DATETIME_UT` | Date and time in UTC.                                                                                                                                                                                                                         |
| `DATETIME`    | Local time, calculated using the longitude of the observatory and rounded to the nearest minute.                                                                                                                                              |
| `H_VALUE`     | INTERMAGNET value of the geomagnetic horizontal component, H.                                                                                                                                                                                 |
| `SPLINE`      | Solar regular variation, denoted as $S_R$, computed using the LDi method. If the day is identified as quiet, this is the $S_R$ curve for that day. If the day is identified as disturbed, this is the $S_R$ curve from the nearest quiet day. |
| `BASELINE`    | Baseline level computed using the LDi method.                                                                                                                                                                                                 |
| `LDi`         | Local Disturbance index value. It represents the resulting geomagnetic perturbation, computed as: $LDi = H - (BASELINE + S_R)$.                                                                                                               |
| `LCi`         | Time derivative of `LDi`.                                                                                                                                                                                                                     |

### File naming convention

Files are named using the following structure:

```text
stationYYYYMMDD_quality.csv
```

For example:

```text
sfs20240501_q.csv
```

where:

* `sfs` is the observatory/station code.
* `20240501` is the date in `YYYYMMDD` format.
* `_q` indicates the data-quality category.

The possible data-quality suffixes are:

| Suffix | Meaning                      |
| ------ | ---------------------------- |
| `_r`   | Reported data                |
| `_p`   | Adjusted or provisional data |
| `_q`   | Quasi-definitive data        |
| `_d`   | Definitive data              |

For the detailed definition of each data-quality category, data format, and naming convention, see the INTERMAGNET Technical Reference Manual linked above.
