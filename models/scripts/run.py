import argparse
import os

import scripts.utils as utils
from datetime import datetime
from scripts.common_features import get_newest_data
from scripts.predict import predict
from scripts.cv_predict import run_cv
from scripts.analyze import analyze

def main(data = None, env = None, n_splits=5, test_size=0.2, seed=42, thresh=0.5, models_dir = None):
    if data is None:
        data = get_newest_data()
    if env is None:
        print("[INFO]: No environment specified. Running for ALL environments.")

    EXP_PATH = prepare_directory(data, env)
    print("[INFO]: Running pipeline for:", EXP_PATH)

    print("[INFO]: Running cross-validation....")
    try:
        run_cv(EXP_PATH, data, n_splits, seed)
    except Exception as e:
        print(f"[ERROR]: Cross-validation failed: {e}")
        print("[INFO]: Continuing to prediction stage....")
    print(f"[INFO]: Cross-validation Results saved to: {EXP_PATH}/cv/")

    print("[INFO]: Running prediction....")
    try :
        skipped = predict(EXP_PATH, data, env, test_size=test_size, seed=seed, thresh=thresh, models_dir=models_dir)
    except Exception as e:
        print(f"[ERROR]: Prediction failed: {e}")
        print("[INFO]: Pipeline terminated.")
        return

    print("[INFO]: Running Analysis....")
    try:
        analyze(EXP_PATH, data, env, skipped)
        print(f"[INFO]: Analysis Results saved to: {EXP_PATH}/analysis/")
    except Exception as e:
        print(f"[ERROR]: Analysis failed: {e}")
        print("[INFO]: Pipeline terminated.")
        return

    print("[INFO]: Pipeline completed successfully.")
    print("[INFO]: All Results saved to:", EXP_PATH, "/")
    print(f"EXPERIMENT_DIR={EXP_PATH}")


def prepare_directory(data_csv, env):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    version = data_csv.split("_")[-1].split(".")[0]
    env_tag = env if env is not None else "All"
    experiment_name = f"exp_{env_tag}_{version}_{timestamp}"
    user = os.environ.get("USER")
    osv = os.environ.get("OS")
    print("[INFO]: Creating experiment directory:", experiment_name)
    js = {"name": f"{env_tag} on dataset", "user": user, "os": osv ,"data_csv": data_csv, "env": env_tag, "timestamp": timestamp, "version": version, "experiment_name": experiment_name}
    utils.create_sub_paths(experiment_name, js)
    return experiment_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-stage prediction and analysis pipeline for ML-Agents Results.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data used as test set (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splitting (default: 42).")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="Threshold for pred_reach from p_reach (default: 0.5).")
    parser.add_argument("--env", dest="env", type=str, default=None, help="Environment to use (default: None).")
    parser.add_argument("--models-dir", dest="models_dir", type=str, default=None,
                        help="Directory containing standard model names: `logistic_reach_model.joblib`, `linear_steps_model.joblib`, `linear_time_model.joblib`.")
    parser.add_argument("--data-csv", type=str, dest="data_csv", default=None,
                        help="Path to CSV containing data. Defaults to latest data in ../data/normalized/.")
    args = parser.parse_args()
    main(test_size=args.test_size, seed=args.seed, thresh=args.thresh, models_dir=args.models_dir, data=args.data_csv,
         env=args.env)