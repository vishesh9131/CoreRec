# load_balancing implementation
import logging
from queue import Queue
from threading import Thread
import time
import threading
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoadBalancing:
    def __init__(self, num_workers=4):
        """
        Initializes the LoadBalancing with a specified number of worker threads.

        Parameters:
        - num_workers (int): The number of worker threads to spawn.
        """
        self.num_workers = num_workers
        self.task_queue = Queue()
        self.results = []
        self.threads = []
        self._init_workers()
        logger.info(f"LoadBalancing initialized with {self.num_workers} workers.")

    def _init_workers(self):
        """
        Initializes worker threads that continuously process tasks from the queue.
        """
        for i in range(self.num_workers):
            thread = Thread(target=self._worker, name=f"Worker-{i+1}", daemon=True)
            thread.start()
            self.threads.append(thread)
            logger.debug(f"Started {thread.name}.")

    def _worker(self):
        """
        Worker thread that processes tasks from the queue.
        """
        while True:
            func, args, kwargs = self.task_queue.get()
            if func is None:
                # Sentinel found, terminate the thread
                logger.debug(f"{threading.current_thread().name} received sentinel. Exiting.")
                break
            try:
                result = func(*args, **kwargs)
                self.results.append(result)
                logger.debug(f"{threading.current_thread().name} processed a task with result: {result}")
            except Exception as e:
                logger.error(f"Error processing task: {e}")
            finally:
                self.task_queue.task_done()

    def add_task(self, func, *args, **kwargs):
        """
        Adds a new task to the queue.

        Parameters:
        - func (callable): The function to execute.
        - *args: Positional arguments for the function.
        - **kwargs: Keyword arguments for the function.
        """
        self.task_queue.put((func, args, kwargs))
        logger.debug(f"Added task {func.__name__} to the queue.")

    def get_results(self):
        """
        Waits for all tasks to be processed and returns the results.

        Returns:
        - list: A list of results from all tasks.
        """
        self.task_queue.join()
        return self.results

    def shutdown(self):
        """
        Shuts down all worker threads gracefully by sending sentinel tasks.
        """
        for _ in self.threads:
            self.task_queue.put((None, (), {}))  # Sentinel
        for thread in self.threads:
            thread.join()
            logger.debug(f"{thread.name} has terminated.")
        logger.info("LoadBalancing has been shutdown.")
