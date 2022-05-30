import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path

from src.lp_svr import svr_experiment

parser = ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--noise", type=float, default=0.0)
parser.add_argument("--p", type=float)
parser.add_argument("--n_splits", type=int, default=0)
parser.add_argument("--random_split_size", type=float, default=0.0)
parser.add_argument("--normalize_data", type=lambda x: bool(int(x)), default=True)
parser.add_argument("--normalize_labels", type=lambda x: bool(int(x)), default=True)
parser.add_argument("--solver", type=str, default="MOSEK")
parser.add_argument("--save_dir", type=Path, default=Path(".") / "out" / "regression")
parser.add_argument("--random_state", type=int, default=None)

args = parser.parse_args()

res = svr_experiment(
    args.dataset,
    args.p,
    args.noise,
    args.solver,
    args.n_splits,
)

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
out_file = Path(args.save_dir) / "res.pkl"
with open(out_file, "wb") as file:
    pkl.dump(res, file)
