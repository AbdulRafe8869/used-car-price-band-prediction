"""Run the whole project, start to finish, with one command.

    python -m src.run_all                 # train -> evaluate -> experiments -> figures
    python -m src.run_all --fast          # tiny grids (smoke test, ~1 minute)
    python -m src.run_all --skip-experiments

The full run takes roughly 15-20 minutes on a 2-core laptop (the Random Forest
grid search is most of it).  Afterwards:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import argparse
import time

from . import evaluate, experiments, make_figures, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="tiny hyper-parameter grids (smoke test only)")
    ap.add_argument("--skip-experiments", action="store_true", help="skip the bonus experiments")
    args = ap.parse_args()

    t0 = time.time()
    print("== 1/4 train ==", flush=True)
    train.main(fast=args.fast)
    print("== 2/4 evaluate ==", flush=True)
    evaluate.print_summary(evaluate.evaluate_all())
    if not args.skip_experiments:
        print("== 3/4 experiments ==", flush=True)
        experiments.main()
    print("== 4/4 figures ==", flush=True)
    make_figures.main()
    print(f"all done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
