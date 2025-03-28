# Path to your requirements.txt file
$RequirementsFile = "requirements.txt"

# Read each line of the file
$Dependencies = Get-Content $RequirementsFile

foreach ($Dependency in $Dependencies) {
    # Remove comments and trim whitespace
    $Dependency = ($Dependency -split '#')[0].Trim()
    if (-not $Dependency) { continue }

    # Extract the package name by removing common version specifiers
    $PackageName = $Dependency -split '==|>=|<=|~=|>|<' | Select-Object -First 1

    if ($PackageName) {
        Write-Host "Adding $PackageName to Poetry..."
        poetry add $PackageName
    }
}
