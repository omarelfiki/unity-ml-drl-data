# Updating Shared Results

All training runs automatically append a new entry to:

data/collected_results.csv

These files are shared by the whole team — so follow this workflow to keep everyone’s results organized and avoid data loss.

---

## 1. Before running a new training
Always pull the latest data first:

```bash
git pull --rebase
```
This ensures your local results file is up to date.

## 2. After training completes

The script automatically adds your new results to the files above.

Commit and push your updates:

```bash
git add data/collected_results.csv
git commit -m "Add results for Run [YourRunID] ([Environment/ConfigName])"
git push
```

## 3. If you see a merge conflict

If Git reports something like:
```abpublidot
CONFLICT (content): Merge conflict in data/collected_results.csv
```

Open the file, keep both lines (each is a valid run), then:
```abpublidot
git add data/collected_results.csv
git commit -m "Resolve CSV merge conflict"
git push
```

> **Never delete or reorder rows** in the CSV
> 
> **Always pull before** running new trainings
