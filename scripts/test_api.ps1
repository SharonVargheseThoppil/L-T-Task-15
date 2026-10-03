Write-Host "=============================================="
Write-Host "TASK 15 - FLASK API TEST"
Write-Host "=============================================="

$API_URL = "http://127.0.0.1:5000"
$IMAGE_PATH = "D:\Task15_End_to_End_DeepLearning\test_images\test_image.jpg"

Write-Host ""
Write-Host "1. Testing root endpoint..."
Invoke-RestMethod "$API_URL/"

Write-Host ""
Write-Host "2. Testing health endpoint..."
Invoke-RestMethod "$API_URL/health"

Write-Host ""
Write-Host "3. Testing prediction endpoint..."

curl.exe -X POST `
    -F "image=@$IMAGE_PATH" `
    "$API_URL/predict"

Write-Host ""
Write-Host "=============================================="
Write-Host "API TEST COMPLETED"
Write-Host "=============================================="