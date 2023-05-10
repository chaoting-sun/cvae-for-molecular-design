from pathos.multiprocessing import ProcessingPool as Pool


def mapper(fn, obj, n_jobs):
    if n_jobs == 1:
        res = list(map(fn, obj))
    else:
        with Pool(n_jobs) as pool:
            res = pool.map(fn, obj)
    return res