# Kubernetes Deployment for LLM Fine-Tuning Expert Suite

Production-grade Kubernetes deployment configuration for the LLM Fine-Tuning Expert Suite.

## 📋 Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- NVIDIA GPU nodes (for training/inference)
- Storage provisioner for PVCs
- LoadBalancer controller

## 🚀 Deployment Steps

### 1. Create Namespace
```bash
kubectl apply -f namespace.yaml
```

### 2. Deploy Redis (Cache)
```bash
kubectl apply -f redis-deployment.yaml
```

### 3. Deploy PostgreSQL (Database)
```bash
kubectl apply -f postgres-deployment.yaml
```

### 4. Deploy MLflow (Experiment Tracking)
```bash
kubectl apply -f mlflow-deployment.yaml
```

### 5. Create ConfigMaps and Secrets
```bash
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
```

### 6. Deploy Application
```bash
kubectl apply -f deployment.yaml
```

### 7. Create Service
```bash
kubectl apply -f service.yaml
```

### 8. Configure Auto-scaling
```bash
kubectl apply -f hpa.yaml
```

## 🔧 Configuration

### Environment Variables
Edit `configmap.yaml` to configure:
- `MODEL_PATH`: Path to model files
- `DEVICE`: cuda/cpu
- `LOG_LEVEL`: INFO/DEBUG/WARNING
- `BATCH_SIZE`: Batch size for inference
- `QUANTIZATION`: 4bit/8bit/fp8

### Resource Limits
Adjust resource requests/limits in `deployment.yaml` based on:
- Available GPU memory
- Model size
- Expected load

## 📊 Monitoring

### Check Pod Status
```bash
kubectl get pods -n llm-finetuning
kubectl logs -f deployment/llm-finetuning-app -n llm-finetuning
```

### Monitor Metrics
```bash
kubectl top pods -n llm-finetuning
kubectl top nodes
```

### Access Services
```bash
kubectl get svc -n llm-finetuning
kubectl port-forward svc/llm-finetuning-service 8080:80 -n llm-finetuning
```

## 🔒 Security

### Secrets Management
Create secrets for sensitive data:
```bash
kubectl create secret generic database-secret \
  --from-literal=url="postgresql://user:pass@host:5432/db" \
  -n llm-finetuning
```

### Network Policies
Apply network policies to restrict traffic:
```bash
kubectl apply -f network-policy.yaml
```

## 📈 Scaling

### Manual Scaling
```bash
kubectl scale deployment llm-finetuning-app --replicas=5 -n llm-finetuning
```

### Auto-scaling
The HPA automatically scales based on CPU/memory usage (2-10 replicas).

## 🔄 Updates

### Rolling Update
```bash
kubectl set image deployment/llm-finetuning-app \
  llm-finetuning=llm-finetuning:v2 \
  -n llm-finetuning
```

### Rollback
```bash
kubectl rollout undo deployment/llm-finetuning-app -n llm-finetuning
```

## 🐛 Troubleshooting

### Check Events
```bash
kubectl describe pod <pod-name> -n llm-finetuning
```

### Debug Containers
```bash
kubectl exec -it <pod-name> -n llm-finetuning -- /bin/bash
```

### View Logs
```bash
kubectl logs -f deployment/llm-finetuning-app -n llm-finetuning
```

## 📝 Notes

- GPU nodes required for training workloads
- Ensure sufficient storage for model files
- Configure resource limits based on cluster capacity
- Monitor GPU utilization with nvidia-dcgm-exporter
- Set up Prometheus/Grafana for comprehensive monitoring
