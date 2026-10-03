Write-Host "=============================================="
Write-Host "TASK 15 - KUBERNETES VALIDATION"
Write-Host "=============================================="

Write-Host ""
Write-Host "1. Cluster Information"
kubectl cluster-info

Write-Host ""
Write-Host "2. Nodes"
kubectl get nodes

Write-Host ""
Write-Host "3. Deployments"
kubectl get deployments

Write-Host ""
Write-Host "4. Pods"
kubectl get pods -o wide

Write-Host ""
Write-Host "5. Services"
kubectl get services

Write-Host ""
Write-Host "6. API Pods"
kubectl get pods -l app=task15-api

Write-Host ""
Write-Host "7. Frontend Pods"
kubectl get pods -l app=task15-frontend

Write-Host ""
Write-Host "8. API Deployment"
kubectl get deployment task15-api

Write-Host ""
Write-Host "9. Frontend Deployment"
kubectl get deployment task15-frontend

Write-Host ""
Write-Host "10. API Logs"
kubectl logs -l app=task15-api --tail=20

Write-Host ""
Write-Host "11. Frontend Logs"
kubectl logs -l app=task15-frontend --tail=20

Write-Host ""
Write-Host "12. Resource Usage"

kubectl top pods

Write-Host ""
Write-Host "=============================================="
Write-Host "KUBERNETES VALIDATION COMPLETED"
Write-Host "=============================================="