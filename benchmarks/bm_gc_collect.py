import gc
import time
from statistics import mean

number_of_iterations = 20
number_of_gc_iterations = 50

deltas = []

gc.disable()
gc.collect()
for kk in range(number_of_iterations):
    t0 = time.perf_counter()
    for jj in range(number_of_gc_iterations):
        gc.collect()
    dt = time.perf_counter() - t0
    deltas.append(dt)
print(f"time per collection: mean {1e3 * mean(deltas) / number_of_iterations:.3f} [ms], min {1e3 * min(deltas) / number_of_iterations:.3f} [ms]")

sets = [frozenset([ii]) for ii in range(10_000)]
deltas = []
print("---")
gc.disable()
gc.collect()
for kk in range(number_of_iterations):
    t0 = time.perf_counter()
    for jj in range(number_of_gc_iterations):
        gc.collect()
    dt = time.perf_counter() - t0
    deltas.append(dt)
print(f"time per collection: mean {1e3 * mean(deltas) / number_of_iterations:.3f} [ms], min {1e3 * min(deltas) / number_of_iterations:.3f} [ms]")

#%% Show statistics of frozen containers

gc.collect()

def candidate(obj):
    return all(not gc.is_tracked(x) for x in obj)

for immutable_type in (tuple, frozenset):
    number_of_objects_tracked = 0
    number_of_candidates = 0
    number_of_immutable_candidates = 0

    for obj in gc.get_objects():
        number_of_objects_tracked += 1
        if type(obj) is immutable_type:
            number_of_candidates += 1
            # print(f"{type(obj)} = {obj}")
            if candidate(obj):
                number_of_immutable_candidates += 1

    print(f"type {immutable_type}")
    print(f"  {number_of_objects_tracked=}")
    print(f"  {number_of_candidates=}")
    print(f"  {number_of_immutable_candidates=}")
