$passwords = @("postgres", "admin", "password", "1234", "root", "")
foreach ($pw in $passwords) {
    $env:PGPASSWORD = $pw
    $result = psql -U postgres -h localhost -c "\q" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS! Password is: '$pw'"
    }
}
