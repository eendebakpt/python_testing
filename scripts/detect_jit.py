import time
import types
import _opcode


def best_case_func(x):
    for ii in range(10_000):
        x = x + 1.1 * x
        x = x / 2.3
        x = 3 * x
    return x


def workload_func():
    def inner():
        [y for y in [x for x in range(2000)]]

    [inner() for x in range(50)]


def run():
    def inner():
        [workload_func() for x in range(5)]

    return inner


def main():
    # use fast local variable
    local_func = run()

    # Warm up?
    [local_func() for x in range(100)]

    t0 = time.perf_counter()
    [local_func() for x in range(100)]
    dt = time.perf_counter() - t0
    print(f"Took {dt} seconds.")

    t0 = time.perf_counter()
    [best_case_func(x / 100) for x in range(1_000)]
    dt = time.perf_counter() - t0
    print(f"best_case_func tok {dt} seconds.")

    print(f"{is_jitted(workload_func)=}")
    print(f"{is_jitted(best_case_func)=}")
    print(f"{is_jitted(run)=}")


def is_jitted(f: types.FunctionType) -> bool:
    for i in range(0, len(f.__code__.co_code), 2):
        try:
            print(f"Valid executor for {f.__name__}: {_opcode.get_executor(f.__code__, i).is_valid()}")
        except RuntimeError:
            # This isn't a JIT build:
            return False
        except ValueError:
            # No executor found:
            continue
        return True
    return False


if __name__ == "__main__":
    main()
