import argparse
import os

from functools import partial

import multiprocessing as mp
import numpy as np
import pandas as pd

import online_kde
import opes
import simulate

parser = argparse.ArgumentParser()
parser.add_argument("--np", help="number of processes", type=int, default=40)
parser.add_argument("--method", help="method to use", type=str, default="opes")
parser.add_argument("--unreweighted", help="use reweighted FES", action="store_true")
parser.add_argument("--uncorrected", help="use fes correction", action="store_true")
parser.add_argument("--bounded", help="unbounded kernels", action="store_true")
parser.add_argument("--uncompressed", help="compression of grid", action="store_true")
parser.add_argument("--incomingbw", help="use incoming bandwidth", action="store_true")
parser.add_argument("--frombias", help="use bias instead of PDF", action="store_true")
args = parser.parse_args()

opes.REWEIGHTED_FES = not args.unreweighted
opes.CORRECTED_OPES_EXPLORE = not args.uncorrected
opes.USE_BIAS_INSTEAD_OF_PDF = args.frombias
online_kde.BOUNDED_KERNELS = args.bounded
online_kde.KEEP_GRID_UNCOMPRESSED = args.uncompressed
online_kde.USE_EXISTING_BANDWIDTHS = not args.incomingbw

num_processes = args.np
method = args.method
directory = method

if args.unreweighted:
    directory += "_unreweighted"
if args.uncorrected:
    directory += "_uncorrected"
if args.frombias:
    directory += "_frombias"
if args.bounded:
    directory += "_bounded"
if args.uncompressed:
    directory += "_uncompressed"
if args.incomingbw:
    directory += "_incomingbw"

program = partial(simulate.muller_brown, method=method, directory=directory)

if __name__ == "__main__":
    print("Directory:", directory)
    with mp.Pool(processes=num_processes) as pool:
        times = np.array(pool.map(program, range(num_processes)))
    with open(os.path.join(directory, "parameters.txt"), "w") as f:
        f.write(
            f"method={method}\n"
            f"simulate.VARIANCE_PACE={simulate.VARIANCE_PACE}\n"
            f"opes.REWEIGHTED_FES={opes.REWEIGHTED_FES}\n"
            f"opes.CORRECTED_OPES_EXPLORE={opes.CORRECTED_OPES_EXPLORE}\n"
            f"opes.USE_BIAS_INSTEAD_OF_PDF={opes.USE_BIAS_INSTEAD_OF_PDF}\n"
            f"online_kde.BOUNDED_KERNELS={online_kde.BOUNDED_KERNELS}\n"
            f"online_kde.KEEP_GRID_UNCOMPRESSED={online_kde.KEEP_GRID_UNCOMPRESSED}\n"
            f"online_kde.USE_EXISTING_BANDWIDTHS={online_kde.USE_EXISTING_BANDWIDTHS}\n"
            f"execution_times={times.mean():.3f} +/- {times.std():.3f} s\n"
        )
    print(f"Done in {times.mean():.3f} +/- {times.std():.3f} s")

    dataframes = [
        pd.read_csv(f"{directory}/{method}_{index:02d}.csv.gz")
        for index in range(num_processes)
    ]
    for index, df in enumerate(dataframes):
        df["run"] = index
        df["xmin"] = np.minimum.accumulate(df["x"])
        df["xmax"] = np.maximum.accumulate(df["x"])
    frame = pd.concat(dataframes)
    means = frame.groupby(["time"]).mean().reset_index()
    stdevs = frame.groupby(["time"]).std().reset_index()
    for col in ["variance", "z", "n", "delta_f"]:
        means[f"stdev[{col}]"] = stdevs[col]
    means.to_csv(f"{directory}/{method}_means.csv.gz", float_format="%.6g", index=False)
