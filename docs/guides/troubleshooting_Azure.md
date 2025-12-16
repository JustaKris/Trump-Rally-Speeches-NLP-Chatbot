# Azure App Service Deployment Troubleshooting

## Issue: RAG Service Returns 503 "RAG service not initialized"

### Symptoms

- Health check (`/health`) returns 200 OK
- Homepage (`/`) works fine
- RAG endpoints (`/rag/ask`, `/rag/search`) return 503
- Error message: "RAG service not initialized. Please try again later."

### Root Cause

RAG service initialization failed during container startup. The app continues to run but RAG functionality is disabled.

### Diagnostic Steps

1. **Check Startup Logs**

   Look for the critical error block in your Azure App Service logs:

   ```text
   ✗ CRITICAL: RAG service initialization failed!
   Error type: <exception type>
   Error message: <details>
   ```

2. **Access Diagnostics Endpoint**

   Visit: `https://your-app.azurewebsites.net/diagnostics`

   This shows:
   - Environment variables status
   - API key configuration (length, not value)
   - Data directory existence
   - Speech file count
   - Service initialization status

3. **Check Health Endpoint**

   Visit: `https://your-app.azurewebsites.net/health`

   Should show:

   ```json
   {
     "status": "degraded",
     "services": {
       "rag_service": false,  ← This indicates the problem
       "llm_configured": true/false
     }
   }
   ```

### Common Causes & Solutions

#### 1. Missing or Invalid LLM API Key

**Symptoms:**

- `llm_configured: false` in `/health`
- `LLM_API_KEY` not set in diagnostics

**Solution:**

```bash
# Add to App Service Configuration > Application Settings
LLM_API_KEY=your_actual_api_key_here
LLM_PROVIDER=gemini  # or openai, anthropic
```

After adding, **restart** the app service.

#### 2. Missing Data Directories

**Symptoms:**

- Diagnostics shows `chromadb_exists: false` or `speeches_exists: false`
- Startup logs show file not found errors

**Solution:**
The data directories should be baked into your Docker image. Verify Dockerfile includes:

```dockerfile
COPY data/ ./data/
```

Rebuild and push the image:

```powershell
docker build -t your-registry/trump-speeches-nlp-chatbot:prod .
docker push your-registry/trump-speeches-nlp-chatbot:prod
```

#### 3. Model Download Failures

**Symptoms:**

- Logs show HuggingFace download errors
- Timeout errors during startup
- Models downloading at runtime instead of being cached

**Solution:**
Models should be pre-downloaded during Docker build. Check Dockerfile has:

```dockerfile
# Should download models as appuser (line ~105)
USER appuser
ENV ENVIRONMENT=production
RUN python scripts/download_models.py
```

If missing, rebuild with the fixed Dockerfile from this repo.

#### 4. Wrong Environment Configuration

**Symptoms:**

- App uses `ENVIRONMENT=staging` or `development` when you expected `production`

**Solution:**

```bash
# Set in App Service Configuration
ENVIRONMENT=production
```

Restart the app service.

#### 5. Memory/Resource Constraints

**Symptoms:**

- OOMKilled errors in logs
- Container restarts frequently
- Models load partially then fail

**Solution:**
Increase App Service Plan resources. Recommended minimum:

- **Memory**: 2GB (4GB preferred)
- **CPU**: 1 vCore (2 vCores preferred)

ML models require ~2GB memory total:

- FinBERT: 438MB
- RoBERTa emotion: 329MB
- MPNet embeddings: 437MB
- Cross-encoder: 91MB
- ChromaDB + Python overhead: ~700MB

### Quick Fix Checklist

- [ ] Verify `LLM_API_KEY` is set in App Service Configuration
- [ ] Check `/diagnostics` endpoint for missing paths/files
- [ ] Review startup logs for specific error messages
- [ ] Ensure Docker image includes pre-downloaded models
- [ ] Confirm App Service has sufficient memory (2GB+)
- [ ] Restart App Service after config changes

### Still Not Working?

Check the detailed startup logs for the specific error:

```text
✗ CRITICAL: RAG service initialization failed!
Error type: ValueError
Error message: Invalid API key for Gemini provider
```

The error type and message will point to the exact issue.

### Testing Locally First

Before deploying to Azure, test the exact production configuration locally:

```powershell
# Use production environment
$env:ENVIRONMENT="production"

# Test with Docker (same as Azure runs)
docker build -t test-image .
docker run --rm -it -p 8000:8000 --env-file .env test-image

# Access diagnostics
curl http://localhost:8000/diagnostics
curl http://localhost:8000/health
curl -X POST http://localhost:8000/rag/ask -H "Content-Type: application/json" -d '{"question": "test"}'
```

If it works locally with `ENVIRONMENT=production`, but fails in Azure, the issue is Azure-specific (likely environment variables or resource constraints).
