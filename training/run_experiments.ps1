# run_experiments.ps1

# Activate your virtual environment first if you haven't already!
# .\venv\Scripts\Activate.ps1

$ConfigPath = "configs_official/Crawler.yaml"
$EnvPath = "C:\Users\larry\Uni_Work\Year_2\AI_ML_project\unity-ml-drl-data\unity\CrawlerBuild\MLagentstest.exe"
$BaseRunID = "C_LF_Crawler"

# Loop from seed 13 to 18
for ($i = 39; $i -le 40; $i++) {
    $RunID = "${BaseRunID}_${i}"

    Write-Host "----------------------------------------------------------------"
    Write-Host "Starting Experiment: Seed $i (Run ID: $RunID)"
    Write-Host "----------------------------------------------------------------"

    # Run the command and wait for it to complete
    python -m scripts.train_model --config $ConfigPath --run-id $RunID --headless $EnvPath --seed $i --no-thresholds

    Write-Host "Experiment $i completed."
    Start-Sleep -Seconds 5 # Short pause between runs to ensure file locks clear
}

Write-Host "All experiments finished!"
