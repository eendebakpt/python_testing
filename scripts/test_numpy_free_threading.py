import copy
import unittest
from threading import Barrier, Thread

import numpy as np
from test.support import threading_helper

threading_helper.requires_working_threading(module=True)


class EnumerateThreading(unittest.TestCase):
    @threading_helper.reap_threads
    def test_threading(self):
        number_of_threads = 10
        number_of_iterations = 20
        barrier = Barrier(number_of_threads)

        def work(x, dtypes):
            barrier.wait()
            for _ in range(1000):
                _ = np.cos(x)
                for d in dtypes:
                    new_dtype = copy.copy(d)
                    old_dtype = x.dtype
                    x.dtype = new_dtype
                    del old_dtype

        for it in range(number_of_iterations):
            x = np.array([1.0, 2.0])
            y = np.array([1.0, 2.0])
            z = np.array([1, 2])
            dtypes = [y.dtype, z.dtype]

            worker_threads = []
            for ii in range(number_of_threads):
                worker_threads.append(Thread(target=work, args=[x, dtypes]))

            with threading_helper.start_threads(worker_threads):
                pass

            barrier.reset()


if __name__ == "__main__":
    unittest.main()
