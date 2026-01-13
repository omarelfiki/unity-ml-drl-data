import json
from pathlib import Path
from joblib import dump

def dump_and_save(ts, out, clf, m_steps, m_time, metrics):
    print("[INFO]: Saving models and results....")
    # Save models for later inference
    models_dir = Path("collected_models")
    models_dir.mkdir(parents=True, exist_ok=True)

    version_dir = models_dir / f"version_{ts}"
    version_dir.mkdir(parents=True, exist_ok=True)

    save_errors = {}

    csv_path = version_dir / "models_prediction.csv"
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
        target = version_dir / name
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

    meta_path = version_dir / "predict_metadata.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        print(f"[ERROR]: failed to write metadata: {e}")

    print(f"\n[INFO]: Saved results and models under: collected_models/version_{ts}")
    if save_errors:
        print(f"[WARNING]: Some save operations failed or were skipped: {json.dumps(save_errors, indent=2)}")
