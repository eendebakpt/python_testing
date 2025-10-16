import gc
import time

gc.collect()
gc.disable()
for kk in range(4):
    number_of_iterations = 20
    t0 = time.perf_counter()
    for jj in range(number_of_iterations):
        gc.collect()
    dt = time.perf_counter() - t0

    print(f"time per collection: {1e3 * dt / number_of_iterations:.3f} [ms]")

"""
time per collection: 0.487 [ms]
time per collection: 0.428 [ms]
time per collection: 0.413 [ms]
time per collection: 0.484 [ms]


"""
# %%
if 0:
    import gc

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
                print(f"{type(obj)} = {obj}")
                if candidate(obj):
                    number_of_immutable_candidates += 1

        print(f"type {immutable_type}")
        print(f"  {number_of_objects_tracked=}")
        print(f"  {number_of_candidates=}")
        print(f"  {number_of_immutable_candidates=}")
