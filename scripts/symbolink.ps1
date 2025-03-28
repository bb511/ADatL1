param (
    [string]$TargetPath = "E:\ADatL1",
    [string]$LinkPath = "C:\Users\victo\Documents\ETH\ADatL1\data"
)

# Check if the target path exists
if (-Not (Test-Path $TargetPath)) {
    Write-Host "Error: Target path '$TargetPath' does not exist." -ForegroundColor Red
    exit 1
}

# Check if the link path exists and remove it if it's a directory
if (Test-Path $LinkPath) {
    Write-Host "Warning: Link path '$LinkPath' already exists. Deleting it..." -ForegroundColor Yellow
    Remove-Item $LinkPath -Recurse -Force
}

# Create symbolic link
Write-Host "Creating symbolic link from '$LinkPath' to '$TargetPath'..." -ForegroundColor Green
cmd /c mklink /D "$LinkPath" "$TargetPath"

# Verify success
if (Test-Path $LinkPath) {
    Write-Host "Symbolic link created successfully!" -ForegroundColor Green
} else {
    Write-Host "Failed to create symbolic link." -ForegroundColor Red
}
