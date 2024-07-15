from functools import partial
import multiprocessing as mp

import simulate

simulate.VARIANCE_PACE = 1

np = 40
method = "opes"
directory = f"varfreq{simulate.VARIANCE_PACE:02d}"

program = partial(simulate.modified_wolfe_quapp, method=method, directory=directory)

if __name__ == "__main__":
    with mp.Pool(processes=np) as pool:
        pool.map(program, range(np))
    print("Done!")
