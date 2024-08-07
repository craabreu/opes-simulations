import argparse
import os

from functools import partial

import multiprocessing as mp
import numpy as np
import pandas as pd

import online_kde
import simulate

parser = argparse.ArgumentParser()
parser.add_argument("--np", help="number of processes", type=int, default=40)
parser.add_argument("--method", help="method to use", type=str, default="opes")
parser.add_argument(
    "--varfreq", help="variance frequency", type=int, default=simulate.VARIANCE_PACE
)
parser.add_argument("--bounded", help="unbounded kernels", action="store_true")
parser.add_argument("--uncompressed", help="compression of grid", action="store_true")
parser.add_argument("--incomingbw", help="use incoming bandwidth", action="store_true")
args = parser.parse_args()

simulate.VARIANCE_PACE = args.varfreq
online_kde.BOUNDED_KERNELS = args.bounded
online_kde.UNCOMPRESSED_KDE = args.uncompressed
online_kde.USE_EXISTING_BANDWIDTHS = not args.incomingbw

num_processes = args.np
method = args.method
directory = method
if method != "metad":
    directory += f"_varfreq{args.varfreq:02d}"
if args.bounded:
    directory += "_bounded"
if args.uncompressed:
    directory += "_uncompressed"
if args.incomingbw:
    directory += "_incomingbw"

program = partial(simulate.modified_wolfe_quapp, method=method, directory=directory)

if __name__ == "__main__":
    print("Directory:", directory)
    with mp.Pool(processes=num_processes) as pool:
        times = np.array(pool.map(program, range(num_processes)))
    with open(os.path.join(directory, "parameters.txt"), "w") as f:
        f.write(
            f"method={method}\n"
            f"simulate.VARIANCE_PACE={simulate.VARIANCE_PACE}\n"
            f"online_kde.BOUNDED_KERNELS={online_kde.BOUNDED_KERNELS}\n"
            f"online_kde.UNCOMPRESSED_KDE={online_kde.UNCOMPRESSED_KDE}\n"
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
    fields = ["variance", "z", "n", "delta_f"]
    if method == "opes-explore":
        fields += ["delta_f_unreweighted", "delta_f_uncorrected", "delta_f_from_bias"]
    for col in fields:
        means[f"stdev[{col}]"] = stdevs[col]
    means.to_csv(f"{directory}/{method}_means.csv.gz", float_format="%.6g", index=False)
