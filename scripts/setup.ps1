# Create folders:
$folders = @("data", "logs", "outputs", "results")

foreach ($folder in $folders) {
    if (!(Test-Path -Path $folder)) {
        New-Item -ItemType Directory -Path $folder
        Write-Host "Created folder: $folder"
    }
    else {
        Write-Host "Folder already exists: $folder"
    }
}
Write-Host "Folders created successfully" -ForegroundColor Green

$envFilePath = ".\.env"
$envContent = @"
PROJECT_ROOT="."
RES_DIR="." # set to the desired location
DATA_DIR="${RES_DIR}/data"
LOG_DIR="${RES_DIR}/logs"
OUTPUT_DIR="${RES_DIR}/outputs"
"@

# Write content to .env file
$envContent | Out-File -FilePath $envFilePath -Encoding utf8
Write-Host ".env file created successfully at $envFilePath" -ForegroundColor Green

Write-Host "All done!"