# Industrial Chain Risk Framework

Python code for calculating network indicators from intermediate-input adjacency matrices and aggregating ten indicators into the Industrial Chain Risk Evaluation Index (ISCREI) using entropy-weighted TOPSIS.

## Files

- `codes/risk_indicators/`: Indicator scripts
- `codes/iscrei/topsis.py`: Entropy-weighted TOPSIS aggregation
- `codes/iscrei/data/`: Previously generated intermediate files and `result.csv`

| Indicator script | Main output columns |
| --- | --- |
| `TBP-Based Indicators.py` | `IMS`, `VSD` |
| `SRPL-Based Indicators.py` | `CCIN_RE`, `CCOUT_RE`, `BISC`, `FISC` |
| `CRC-Based Indicators.py` | `CRC` |
| `CFP-Based Indicators.py` | `CFP` |
| `SWOT-Based Indicators.py` | `SPIN`, `SOIN` |

The scripts also write auxiliary columns. The manuscript names two indicators `CCINRE` and `CCOUTRE`, while the SRPL script writes `CCIN_RE` and `CCOUT_RE`.

## Requirements

Use Python 3 with `numpy`, `pandas`, `tqdm`, and `openpyxl`. The CFP script additionally requires CUDA-enabled `torch` and exits when CUDA is unavailable.

```bash
python -m pip install numpy pandas tqdm openpyxl
```

## Run the code

Set the input and output paths in each script's `__main__` block before running it. The indicator scripts produce Excel files; run `topsis.py` from its own directory to produce `data/result.csv`. The required adjacency matrices and TOPSIS input workbook are not included.
