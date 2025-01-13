## This wrapper will be removed once the entire module becomes installable packagw

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
import axo
import argparse
import yaml

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="Process some configurations.")
    parser.add_argument('--config_path',
        nargs='?',
        default=None,
        help="Path to the configuration file (optional, if not provided runs with default config)."
    )
    parser.add_argument('--experiment_name',
        nargs='?',
        default=None,
        help="Experiment Name"
    )

    args = parser.parse_args()

    if args.config_path is None:
        print("No config found, running with default")
        axo.master.main()
    else:
        slave = yaml.load(open(args.config_path, "r"), Loader=yaml.Loader)
        axo.master.main(slave=slave, experiment_name=args.experiment_name)

print("Success")