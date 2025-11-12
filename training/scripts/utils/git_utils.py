"""Git automation utilities."""
import os
import subprocess
from scripts.config.constants import AUTO_COMMIT_BRANCH

def auto_commit_results(v, commit_message="Auto-update: new training results"):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
    files_to_commit = ["combined_results.csv", "combined_results.json"]

    try:
        # Verify branch is selected branch
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        if branch != AUTO_COMMIT_BRANCH:
            print(f"[WARNING] Current branch is '{branch}', not designated auto commit branch '{AUTO_COMMIT_BRANCH}'")
            print(f"[INFO] To change to auto-commit branch, run 'git checkout {AUTO_COMMIT_BRANCH}'")
            print(f"[WARNING] Aborting auto-commit.")
            return

        # Check for a clean working tree
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        status_lines = status.stdout.strip().splitlines()

        def _should_ignore(line: str) -> bool:
            if not line:
                return True
            # Porcelain format: two status chars + space + path
            xy = line[:2]
            raw_path = line[3:].strip()
            if " -> " in raw_path:
                raw_path = raw_path.split(" -> ", 1)[-1].strip()

            # Ignore untracked files entirely
            if xy == "??":
                return True

            # Allow dataset result files to be dirty (staged or unstaged)
            allowed_dirty = ["data/combined_results.csv", "data/combined_results.json"]
            for allowed in allowed_dirty:
                if allowed in line or allowed in raw_path:
                    return True
            print(f"[WARNING] Unrecognized porcelain line: {line}")
            return False

        dirty_changes = [ln for ln in status_lines if not _should_ignore(ln)]

        if any(ln.strip() for ln in dirty_changes):
            print(
                "[WARNING] Working directory not clean (excluding results files). Please commit or stash your changes before training.")
            print(f"[WARNING] Aborting auto-commit.")
            return
        else:
            if v: print("[INFO] Working directory clean of other changes (excluding results files).")

        # Ensure up to date with origin
        subprocess.run(["git", "fetch", "origin"], check=True)
        local = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        remote = subprocess.check_output(["git", "rev-parse", f"origin/{AUTO_COMMIT_BRANCH}"]).decode().strip()
        if local != remote:
            print(f"[WARNING] Local branch not up to date with origin/{AUTO_COMMIT_BRANCH}. Please pull first.")
            print(f"[WARNING] Aborting auto-commit.")
            return

        # Stage files
        print(f"[INFO] Staging dataset files for commit on branch '{branch}'")
        for file in files_to_commit:
            file_path = os.path.join(data_dir, file)
            if os.path.exists(file_path):
                subprocess.run(["git", "add", file_path], check=True)
            else:
                print(f"[WARNING] File not found: {file}")

        result = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if result.returncode != 0:  # there are changes staged
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            subprocess.run(["git", "push"], check=True)
            print("[INFO] Auto-commit completed successfully.")
        else:
            print("[INFO] No dataset changes to commit.")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Git command failed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")