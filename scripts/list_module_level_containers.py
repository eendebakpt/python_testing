"""
Created on Tue Sep 16 20:30:35 2025

@author: eendebakpt
"""

import importlib
import pkgutil
import sys
from collections import Counter

blacklist = "idlelib.idle"
excluded_submodule_names = "__main__"
search_submodules = 2
mutable_containers = (list, dict, set)
mutable_containers = (list,)
only_private = True


def list_container_types(module, mcc):
    print_module = False
    for a in dir(module):
        if a in ("__all__", "__path__", "__builtins__", "__annotations__", "__conditional_annotations__"):
            # why is __all__ a list and not a tuple?
            continue
        if only_private and not a.startswith("_"):
            continue
        attr = getattr(module, a)
        tp = type(attr)
        if issubclass(tp, mutable_containers):
            if not print_module:
                print(f"{module}:")
                print_module = True
            print(f"  {a}: {tp}")
            mcc.update([tp])


def search_modules(module_names, search_submodules: int, mcc):
    for name in module_names:
        if name in blacklist:
            continue
        try:
            module = importlib.import_module(name)
        except:
            print(f"{name}: error on import")
            module = None
        list_container_types(module, mcc)

        if search_submodules:
            try:
                sub_names = list(z.name for z in pkgutil.iter_modules(module.__path__))
            except Exception:
                sub_names = []
            mm = [name + "." + sub_name for sub_name in sub_names if sub_name not in excluded_submodule_names]
            search_modules(mm, search_submodules - 1, mcc)


mcc = Counter()
module_names = sorted(list(sys.builtin_module_names)) + sorted(list(sys.stdlib_module_names))
# module_names=['_pyrepl']
search_modules(module_names, search_submodules=2, mcc=mcc)

print()
print("number of module level containers by type:")
for key, value in mcc.items():
    print(f"{key}: {value}")


# %%
# m=importlib.import_module('_pyrepl.utils')

# print(m)
