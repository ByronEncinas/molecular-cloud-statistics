import sys
import gc
import numpy as np
a = [1] * 10_000_000  # Large list
b = [1] * 10_000_000  # Large list
c = [1] * 10_000_000  # Large list
d = [1] * 10_000_000  # Large list
f = np.ones((40_000, 2_000))

def get_globals_memory() -> None:
    import sys
    total = 0
    for name, obj in globals().items():
        if name.startswith("__") and name.endswith("__"):
            continue
        try:
            total += sys.getsizeof(obj)
        except TypeError:
            pass 

    gb = total / (1024 ** 3)
    print(f"Memory used by globals: {gb:.6f} GB")


a = [1] * 10_000_000  # large list



print("Memory used by numpy array f:", sys.getsizeof(f)/(1024 ** 3))
print("Memory used by a:", sys.getsizeof(a)/(1024 ** 3))
get_globals_memory()

del a,b,c,d
gc.collect()  # Force cleanup

print("Deleted and collected garbage.")
get_globals_memory()

