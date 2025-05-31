import numpy as np

arr = np.zeros(65521, dtype=np.float16)
arr[:10] = 1
z = np.percentile(arr, 50)
print(z)

input_dtype = np.float16
arr = np.asarray([15.0, 20.0, 35.0, 40.0, 50.0], dtype=input_dtype)
q = 0.4
quantiles = np.array(q, dtype=arr.dtype)
v = np.quantile(arr, q)
print(v)
print(v - 29)


# %%

if 0:
    n = arr.size

    np.int64(n - 1) * np.float16(1.6)

    w = (n - np.int64(1)) * q
    print(f" {w=}")

    q = np.array(q, dtype=np.float16)
    w = (n - np.int64(1)) * q
    print(f" {w=}")

    20 * 0.4 + 35 * 0.6

    # %%
    import numpy as np

    print(f"{np.float16(0.4):.41f}")
    print(f"{np.float64(0.4):.41f}")

    # %%
    import numpy as np

    print(f"{np.__version__=}")

    f16 = np.float16(0.4)
    print(f" {f16=}")
    print(f" {4*f16=} {4*f16=:.10f}")
    print(f" {np.int64(4)*f16=}")

    a16 = np.array([0.4], dtype=np.float16)
    print(f" {a16=}")
    print(f" {4*a16=}")
    print(f" {np.int64(4)*a16=}")  # not equal to 1.6
    print(f" {a16*np.int64(4)=}")  # not equal to 1.6

    print(f"{4 * f16:.41f}")
