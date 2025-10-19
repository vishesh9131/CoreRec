# scalable_algorithms implementation
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScalableAlgorithms:
    def __init__(self, num_workers=None):
        """
        Initializes the ScalableAlgorithms with a specified number of worker processes.

        Parameters:
        - num_workers (int, optional): The number of worker processes to use.
                                        Defaults to the number of CPU cores available.
        """
        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count()
        else:
            self.num_workers = num_workers
        logger.info(f"ScalableAlgorithms initialized with {self.num_workers} workers.")

    def parallel_process(self, function, data, chunksize=1):
        """
        Processes data in parallel using a specified function.

        Parameters:
        - function (callable): The function to apply to each data chunk.
        - data (iterable): The data to process.
        - chunksize (int): The size of each data chunk.

        Returns:
        - list: A list of results after applying the function.
        """
        results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_data = {executor.submit(function, item): item for item in data}
            for future in as_completed(future_to_data):
                item = future_to_data[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(f"Processed item: {item} with result: {result}")
                except Exception as exc:
                    logger.error(f"Item {item} generated an exception: {exc}")
        logger.info("Parallel processing completed.")
        return results

    def map_async(self, function, data):
        """
        Asynchronously maps a function over data using multiprocessing.

        Parameters:
        - function (callable): The function to apply to each data item.
        - data (iterable): The data to process.

        Returns:
        - list: A list of results after applying the function.
        """
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            results = pool.map_async(function, data).get()
        logger.info("Asynchronous mapping completed.")
        return results

    def chunkify(self, data, n_chunks):
        """
        Splits data into specified number of chunks.

        Parameters:
        - data (list): The data to split.
        - n_chunks (int): The number of chunks to create.

        Returns:
        - list of lists: A list containing the data chunks.
        """
        chunk_size = len(data) // n_chunks
        chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks - 1)]
        chunks.append(data[(n_chunks - 1) * chunk_size:])
        logger.info(f"Data split into {n_chunks} chunks.")
        return chunks
