#!/usr/bin/env python3
"""Convenience script to combine valgrind and kcachegrind"""

import glob
import os
import subprocess
import sys

# print(sys.argv)

# sys.argv = ['python' , 'numpy_quick.py']

aa = " ".join(sys.argv[1:])
cmd = f"valgrind --tool=callgrind {aa}"
print(f"running cmd: {cmd}")

v = os.system(cmd)

print(f"result: {v}")

if v == 0:
    # start kcachegrind

    list_of_files = glob.glob("callgrind.out.*")
    if len(list_of_files) == 0:
        print("no callgrind files found")
        exit(0)
    latest_file = max(list_of_files, key=os.path.getctime)
    cmd = f"kcachegrind {latest_file}"
    print(f"running cmd: {cmd}")
    subprocess.Popen(["kcachegrind", latest_file])
