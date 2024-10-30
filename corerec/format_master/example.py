import cr_formatMaster as fm

# fm.detect('src/SANDBOX/datasets/ratings.dat', config=None)
config = {
    'parallel_processing': True,
    'log_level': 'INFO',
    'chunk_size': 10000,
    'missing_value_strategy': 'fill_mean',
    'scaling_method': 'standard',
}
fm.detect('src/SANDBOX/datasets/ratings.dat', config=config)
