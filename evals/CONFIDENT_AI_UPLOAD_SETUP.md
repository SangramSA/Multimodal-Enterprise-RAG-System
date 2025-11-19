# Confident AI Upload Setup Guide

This guide will help you set up automatic uploads of evaluation results to Confident AI.

## Prerequisites

1. **Confident AI Account**: Sign up at [https://www.confident-ai.com](https://www.confident-ai.com)
2. **API Key**: Get your API key from the Confident AI dashboard
3. **Python SSL Certificates**: Fix SSL certificate issues (see below)

## Step 1: Fix SSL Certificates

The SSL certificate errors prevent uploads to Confident AI. Fix them first:

### Option A: Run the Fix Script (Recommended)

```bash
python scripts/fix_ssl_certificates.py
```

This script will:
- Install/upgrade the `certifi` package
- Run the macOS certificate installer (if available)
- Set the `SSL_CERT_FILE` environment variable

### Option B: Manual Fix

```bash
# Install certifi
pip install --upgrade certifi

# Set SSL_CERT_FILE in your .env file
echo "SSL_CERT_FILE=$(python3 -m certifi)" >> .env

# Or run the macOS certificate installer
/Applications/Python\ 3.12/Install\ Certificates.command
```

## Step 2: Set Confident AI API Key

Add your Confident AI API key to your `.env` file:

```bash
# DeepEval expects CONFIDENT_API_KEY (not CONFIDENT_AI_API_KEY)
CONFIDENT_API_KEY=your_api_key_here
```

**Important**: DeepEval uses `CONFIDENT_API_KEY` (not `CONFIDENT_AI_API_KEY`).

## Step 3: Run Evaluation with Automatic Upload

### Method 1: Using `--use-automatic-upload` Flag (Recommended)

This uses DeepEval's native `evaluate()` function which automatically uploads results:

```bash
python evals/run_evaluation.py \
  --squad-samples 1 \
  --docvqa-samples 1 \
  --fleurs-samples 1 \
  --use-automatic-upload
```

### Method 2: Manual Evaluation (Current Default)

The current default method uses `measure()` directly and saves results locally. 
SSL errors will prevent automatic uploads, but results are still saved to `logs/eval_results.json`.

```bash
python evals/run_evaluation.py \
  --squad-samples 1 \
  --docvqa-samples 1 \
  --fleurs-samples 1
```

## Step 4: Verify Upload

After running evaluation with `--use-automatic-upload`, you should see:

1. **Success message**: `✅ Evaluation completed and uploaded to Confident AI`
2. **No SSL errors**: No `SSLCertVerificationError` messages
3. **Confident AI Dashboard**: Check your Confident AI dashboard for the new test run

## Troubleshooting

### SSL Certificate Errors Still Occurring

1. **Verify certifi is installed**:
   ```bash
   python -c "import certifi; print(certifi.where())"
   ```

2. **Set SSL_CERT_FILE explicitly**:
   ```bash
   export SSL_CERT_FILE=$(python3 -m certifi)
   ```

3. **Check Python version**:
   ```bash
   python --version
   ```

4. **Reinstall certificates**:
   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

### API Key Not Working

1. **Verify API key is set**:
   ```bash
   python -c "import os; print('CONFIDENT_API_KEY:', 'SET' if os.getenv('CONFIDENT_API_KEY') else 'NOT SET')"
   ```

2. **Check API key format**: Should be a string, not empty

3. **Verify in Confident AI dashboard**: Make sure the API key is active

### Upload Still Fails

1. **Check network connectivity**: Ensure you can reach `api.confident-ai.com`
2. **Check firewall/proxy**: Corporate networks may block HTTPS connections
3. **Try without automatic upload**: Results are still saved locally even if upload fails

## Differences Between Methods

| Feature | Manual (`measure()`) | Automatic (`evaluate()`) |
|---------|---------------------|--------------------------|
| **Confident AI Upload** | ❌ Manual (blocked by SSL) | ✅ Automatic |
| **Caching** | ✅ Supported | ⚠️ Limited |
| **Parallel Execution** | ✅ Supported | ⚠️ Limited |
| **Fine-grained Control** | ✅ Full control | ⚠️ Less control |
| **Result Format** | ✅ Detailed per-metric | ⚠️ Less detailed |

## Recommendation

- **For local development/testing**: Use manual method (default)
- **For production/CI**: Use `--use-automatic-upload` flag
- **For debugging**: Use manual method to see detailed per-metric scores

## Next Steps

After successful setup:
1. Run a small test: `python evals/run_evaluation.py --squad-samples 1 --use-automatic-upload`
2. Check Confident AI dashboard for results
3. Run full evaluation when ready

## References

- [DeepEval Confident AI Integration](https://deepeval.com/docs/getting-started-rag)
- [Confident AI Documentation](https://www.confident-ai.com/docs)
- [SSL Troubleshooting Guide](../docs/SSL_TROUBLESHOOTING.md)

