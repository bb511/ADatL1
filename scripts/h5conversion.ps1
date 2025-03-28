# # Deactivate the current .venv
# deactivate

# # Activate the datamaker .venv
# ${DATAMAKER_PATH}/.venv/Scripts/activate.ps1

# # Run the h5conversion script with two folders:
# python ${DATAMAKER_PATH}/scripts/h5conversion/plot.py --folder="${PROJECT_ROOT}\data\EphZB_2024E_run381148-381149_all_14_0_7_menuv110_1716739892"
# python ${DATAMAKER_PATH}/scripts/h5conversion/plot.py --folder="${PROJECT_ROOT}\data\L1TNtupleRun3-133xWinter24\SingleNeutrino_E-10-gun"

# # Activate the datamaker .venv
# deactivate

# # Reactivate the current .venv
# ${PROJECT_ROOT}/.venv/Scripts/activate.ps1


# Load environment variables from the .env file
$scriptDir = Get-Location
$envFilePath = Join-Path -Path $scriptDir -ChildPath '.env'


# Current .venv activation
$env:PROJECT_ROOT = $env:PROJECT_ROOT.Trim('"')
$activatePath = Join-Path -Path $env:PROJECT_ROOT -ChildPath '.venv\Scripts\activate.ps1'

# Datamaker .venv activation
$env:DATAMAKER_PATH = $env:DATAMAKER_PATH.Trim('"')
$datamakeractivatePath = Join-Path -Path $env:DATAMAKER_PATH -ChildPath ".venv\Scripts\activate.ps1"

# Check if the .env file exists
if (Test-Path $envFilePath) {
    Get-Content $envFilePath | ForEach-Object {
        $key, $value = $_ -split '=', 2
        [System.Environment]::SetEnvironmentVariable($key.Trim(), $value.Trim(), [System.EnvironmentVariableTarget]::Process)
    }
} else {
    Write-Host "Warning: .env file not found at $envFilePath"
}


if (Test-Path $envFilePath) {
    Get-Content $envFilePath | ForEach-Object {
        $key, $value = $_ -split '=', 2
        [System.Environment]::SetEnvironmentVariable($key.Trim(), $value.Trim(), [System.EnvironmentVariableTarget]::Process)
    }
} else {
    Write-Host "Warning: .env file not found at $envFilePath"
}

# Activate the current .venv (for sanity reasons)
Write-Host "Checking if current virtual environment is activated..."
& $activatePath


# Deactivate the current .venv
Write-Host "Deactivating current virtual environment..."
deactivate

# Activate the datamaker .venv
Write-Host "Activating datamaker virtual environment..."
& $datamakeractivatePath

# Run the h5conversion script with the two folders
Write-Host "Running h5conversion script for EphZB_2024E_run381148..."
python "${env:DATAMAKER_PATH}\scripts\h5conversion\plot" --folder="${env:PROJECT_ROOT}\data\EphZB_2024E_run381148-381149_all_14_0_7_menuv110_1716739892"

Write-Host "Running h5conversion script for L1TNtupleRun3-133xWinter24..."
python "${env:DATAMAKER_PATH}\scripts\h5conversion\plot" --folder="${env:PROJECT_ROOT}\data\L1TNtupleRun3-133xWinter24\SingleNeutrino_E-10-gun"

# Deactivate the datamaker .venv
Write-Host "Deactivating datamaker virtual environment..."
deactivate

# Reactivate the current .venv
Write-Host "Reactivating current virtual environment..."
& $activatePath
