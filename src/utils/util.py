import time


def is_close(a, b, abs_tol=1e-6):
    return abs(a - b) < abs_tol


class Timer:
    """
    Timer context manager to measure the time elapsed in a block of code

    Example:
    ```
    with Timer("Execution time:"):
        # block of code
    ```

    Args:
        - prefix: str, prefix to print before the elapsed time
        - postfix: str, postfix to print after the elapsed time
        - print_func: callable, print function to use, default is `print`

    """

    def __init__(self, prefix="", postfix="", print_func=print) -> None:
        self.prefix = prefix
        self.postfix = postfix
        self.print_func = print_func

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args):
        elapsed_time = time.time() - self.start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
        self.print_func(f"{self.prefix} {time_str} {self.postfix}")
