Note: This repository will remain public until May 1st.
# Energy Benchmarking of Python Logging Libraries

This project measures and compares the **energy consumption** of different Python logging libraries using CodeCarbon.


* Supports multiple experiment modes:

  * `fixed_size`
  * `fixed_rate`
  * `fixed_messages`

## Setup

pip install -r requirements.txt


## Run
cd main

python -m core.run_all_experiments


## Configuration
All experiment settings (sizes, rates, repetitions, etc.) are defined in:

core/config.py

## Output

* Results:
  results/summary_<mode>.csv
  
* Logs:
  logs/<library>/<mode>/<statement>/

