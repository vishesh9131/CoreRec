import formatmaster as fm
    
# fm.detect('data/book_history.dat', config=None)
config = {
    'parallel_processing': True,
    'log_level': 'INFO',
    'chunk_size': 10000,
    'missing_value_strategy': 'fill_mean',
    'scaling_method': 'standard',
}
# fm.cr_formatMaster.detect('data/books.csv', config=config)
