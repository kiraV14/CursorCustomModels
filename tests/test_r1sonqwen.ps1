Write-Host "Testing R1sonQwen Model..." -ForegroundColor Green

$headers = @{
    "Content-Type" = "application/json"
    "Authorization" = "Bearer fake-api-key"
}

$body = @{
    model = "r1sonqwen"
    messages = @(
        @{
            role = "system"
            content = "You are a helpful coding assistant."
        },
        @{
            role = "user"
            content = "Write a Python function to calculate the Fibonacci sequence up to n terms."
        }
    )
    temperature = 0.7
    max_tokens = 1000
} | ConvertTo-Json

Write-Host "Sending request to https://f03a-138-84-36-222.ngrok-free.app/v1/chat/completions" -ForegroundColor Yellow
Write-Host "This may take a while as it needs to call two models in sequence." -ForegroundColor Yellow

$startTime = Get-Date

try {
    $response = Invoke-RestMethod -Uri "https://f03a-138-84-36-222.ngrok-free.app/v1/chat/completions" -Method Post -Headers $headers -Body $body -ContentType "application/json"
    
    $elapsedTime = (Get-Date) - $startTime
    Write-Host "Request completed in $($elapsedTime.TotalSeconds) seconds" -ForegroundColor Green
    
    Write-Host "`n=== Response ===" -ForegroundColor Cyan
    Write-Host "Model: $($response.model)" -ForegroundColor Cyan
    
    if ($response.choices -and $response.choices.Count -gt 0) {
        $content = $response.choices[0].message.content
        Write-Host "`n=== Content ===" -ForegroundColor Cyan
        Write-Host $content -ForegroundColor White
        
        if ($response.usage) {
            Write-Host "`n=== Usage ===" -ForegroundColor Cyan
            Write-Host "Prompt tokens: $($response.usage.prompt_tokens)" -ForegroundColor White
            Write-Host "Completion tokens: $($response.usage.completion_tokens)" -ForegroundColor White
            Write-Host "Total tokens: $($response.usage.total_tokens)" -ForegroundColor White
        }
    }
    else {
        Write-Host "No content in response" -ForegroundColor Red
        Write-Host ($response | ConvertTo-Json -Depth 10) -ForegroundColor Red
    }
}
catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host $_.Exception -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $reader.BaseStream.Position = 0
        $reader.DiscardBufferedData()
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response body: $responseBody" -ForegroundColor Red
    }
}

Write-Host "`nPress any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 