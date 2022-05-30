import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path

from src.lp_svc import svc_experiment

parser = ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--p", type=float)
parser.add_argument("--label_noise", type=float, default=0.0)
parser.add_argument("--n_splits", type=int, default=100)
parser.add_argument("--random_split_size", type=float, default=0.1)
parser.add_argument("--normalize_data", type=lambda x: bool(int(x)), default=True)
parser.add_argument("--solver", type=str, default="MOSEK")
parser.add_argument("--save_dir", type=Path, default=Path(".") / "out" / "classification")
parser.add_argument("--random_state", type=int, default=None)
args = parser.parse_args()

res = svc_experiment(
    args.dataset,
    args.p,
    args.label_noise,
    args.solver,
    args.n_splits,
    args.random_split_size,
    args.normalize_data,
    args.random_state,
)

save_dir = Path(args.save_dir) / args.dataset
save_dir.mkdir(parents=True, exist_ok=True)
out_file = save_dir / "res.pkl"
with open(out_file, "wb") as file:
    pkl.dump(res, file)
