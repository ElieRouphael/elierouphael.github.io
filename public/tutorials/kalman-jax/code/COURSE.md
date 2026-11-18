# Kalman Filters with JAX: the course download

This archive holds the code behind the course at <https://elierouphael.github.io/tutorials/kalman-jax/>: three of
the author's repositories, as the course corrected them, and one script per chapter that prints every number the
chapter quotes. The chapters themselves, with the explanations and the live widgets, are on the website.

`README.md` in this folder belongs to the Kalman-Filter-with-JAX repository; this file is the map of the whole archive.

## What is where

| Folder or file | What it is | Chapter |
|---|---|---|
| `vehicle/` | The kalman_vehicle_tracking repository: a twenty-line NumPy Kalman filter and the tracker that runs it | 01 |
| `kalman_jax/`, `tests/`, `README.md`, `requirements.txt`, `data/README.md` | The Kalman-Filter-with-JAX repository: its package, tests and notes | 02, 03, 04 |
| `Kalman Filter with JAX.py` | The linear, extended and unscented filters on a simulated track | 02 |
| `Kalman Filter with DL.py` | A residual network trained inside the filter | 03 |
| `Air Quality with JAX.py` | Next-hour NO2 forecasts on the UCI Air Quality data | 04 |
| `first-order-c/` | The First-Order-Filter-using-C repository: `TD3_original.c` as first committed, the corrected `TD3.c`, and the recording both read | 05 |
| `part1/` | `j1_numbers.py` to `j6_numbers.py`, one per Part I chapter, and the `requirements.txt` with the versions they were checked with | J1 to J6 |
| `part2/` | `ch01_numbers.py` to `ch05_numbers.py`, one per Part II chapter, and in `part2/data/` the data they need | 01 to 05 |

## Two vehicle data files

`vehicle/data/example_data.csv` is the repository's current file, regenerated with a fixed seed after the review.
Chapter 01 uses the file as first committed, which is shipped as `part2/data/vehicle_example_data_original.csv`.
They are two draws of the same noise, so the numbers differ a little (position RMSE):

| Data file | Raw sensor | Kalman, Q = diag(1, 3) | Kalman, 10 × diag(1, 3) |
|---|---|---|---|
| `vehicle/data/example_data.csv` (regenerated) | 4.890 m | 4.492 m | 2.989 m |
| `part2/data/vehicle_example_data_original.csv` (chapter 01) | 5.003 m | 4.500 m | 2.934 m |

`vehicle/tracker/vehicle_tracker.py` prints the first row's raw and ten-times numbers; `part2/ch01_numbers.py` prints
both rows.

## Running it

From this folder:

```bash
pip install -r requirements.txt
(cd vehicle && python tracker/vehicle_tracker.py)   # chapter 01 (on its regenerated CSV: 4.89 m raw, 2.99 m filtered)
python "Kalman Filter with JAX.py"          # chapter 02: linear KF, EKF, UKF
python "Kalman Filter with DL.py"           # chapter 03: the learned residual
pytest
pip install -r part1/requirements.txt     # Part I: the versions its numbers were checked with
python part1/j3_numbers.py                # Part I: every number on chapter J3 (likewise j1 to j6)
python part2/ch01_numbers.py              # Part II: every number on chapter 01 (numpy; its last section also jax and optax)
python part2/ch02_numbers.py              # Part II: every number on chapter 02 (jax and scipy, as part1/)
python part2/ch03_numbers.py              # Part II: every number on chapter 03 (jax and optax, as part1/; a few minutes)
python part2/ch04_numbers.py data/raw/AirQualityUCI.csv   # Part II: chapter 04 (also pandas; several minutes)
python part2/ch05_numbers.py              # Part II: every number on chapter 05 (jax; compiles the C if gcc is on PATH)
gcc -Wall -Wextra -o td3 first-order-c/TD3.c -lm && (cd first-order-c && ../td3)   # chapter 05
```

`requirements.txt` is the JAX repository's own and accepts any recent JAX. `part1/requirements.txt` pins the
versions the course's numbers were printed with (JAX 0.12, Optax 0.2.8); with another JAX release, printed jaxprs,
error messages and timings can differ slightly. Each script's docstring says what it needs and how long it takes.

Chapter 04 needs the UCI Air Quality dataset, which is not included: download `AirQualityUCI.csv` from
<https://doi.org/10.24432/C59K5F> (CC BY 4.0) into `data/raw/`, then run `python "Air Quality with JAX.py"`.
