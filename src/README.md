# Usage
`python main.py`

`python main.py -b benchmark_config.json`

`python main.py -s system_config.json`

`python main.py -b benchmark_config.json -s system_config.json`

# Components
All components are tied up together with the interface
## Benchmark specification
## Models
## Datasets
## Algorithms
## Evaluation
Evaluation involves two category of metrics. One category of metric is specific to the algorithm and the other category of metric is generic for all of the algorithms and used for benchmarking.
## Scheduler

## TODOs
- [] replace prints with loggers
- [] fix redundancy and code smells in dataset_utils.py
- [] remove all usages/references to the warmup_logs
- [] add named keys functionality in the json file. The experiments folder name will then add things by default and add key-value pair from the json file labeled in some way
- [] 