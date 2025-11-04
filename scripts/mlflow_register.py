#!/usr/bin/env python
from __future__ import annotations

import argparse
from src.tracking.mlflow_utils import start_run, log_params, register_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--model-uri", required=True)
    p.add_argument("--name", required=True)
    args = p.parse_args()

    with start_run(args.experiment, args.run_name):
        log_params({"registered_model": args.name})
        result = register_model(args.model_uri, args.name)
        print(result)


if __name__ == "__main__":
    main()
