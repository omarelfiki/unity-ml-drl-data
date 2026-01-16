import json
from pathlib import Path
from joblib import dump

EXP_DIR = Path("experiments")

def create_sub_paths(experiment_name, js):
    experiment_dir = EXP_DIR / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    js_path = experiment_dir / "metadata.json"
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2)

    CV_DIR = experiment_dir / "cv"
    CV_DIR.mkdir(parents=True, exist_ok=True)

    PRED_DIR = experiment_dir / "predictions"
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    ANALYSIS_DIR = experiment_dir / "analysis"
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    return experiment_dir


def dump_and_save(pred_dir, out, clf, m_steps, m_time, metrics, env_name):
    print("[INFO]: Saving models and results....")
    save_errors = {}

    env_dir = pred_dir / str(env_name)
    env_dir.mkdir(parents=True, exist_ok=True)

    csv_path = env_dir / "models_prediction.csv"
    try:
        out.to_csv(csv_path, index=False)
    except Exception as e:
        save_errors["models_prediction.csv"] = str(e)
        print(f"[ERROR]: failed to write CSV: {e}")

    model_targets = [
        (clf, "logistic_reach_model.joblib"),
        (m_steps, "linear_steps_model.joblib"),
        (m_time, "linear_time_model.joblib"),
    ]
    for model, name in model_targets:
        target = env_dir / name
        if model is None:
            save_errors[name] = "model_missing"
            continue
        try:
            dump(model, target)
        except Exception as e:
            save_errors[name] = str(e)
            print(f"[ERROR]: failed to dump {name}: {e}")

    if save_errors:
        metrics["save_errors"] = save_errors

    meta_path = env_dir / "metadata.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        print(f"[ERROR]: failed to write metadata: {e}")

    print(f"\n[INFO]: Saved results and models under: {env_dir.resolve()}\n")
    if save_errors:
        print(f"[WARNING]: Some save operations failed or were skipped: {json.dumps(save_errors, indent=2)}")