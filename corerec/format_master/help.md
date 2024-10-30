# CoreRec Format Master Plug

## cr_format_master : 
#### This format detection pipeline will be later plugged into corerec's utilities.
`import cr_formatMaster as fm`

`fm.detect(df)`

### we can also pass a config dictionary to the "detect" function.

Example config:
```python
config = {
    'parallel_processing': True,
    'log_level': 'INFO',
    'chunk_size': 10000,
    'missing_value_strategy': 'fill_mean',
    'scaling_method': 'standard',
    'validation_rules': {'max_null_percentage': 0.1},
    'report_format': 'json',
    'log_file': 'pipeline.log',
    'monitoring': True,
    'num_workers': 4,
    'distributed_backend': 'dask',
    'custom_steps': ['step1', 'step2']
    }

```

## ds_format_loader : 
#### This module provides functions to load, preprocess, and validate data from various file formats for use in a recommendation system.


## format_library : 
#### exception handling for various file formats.