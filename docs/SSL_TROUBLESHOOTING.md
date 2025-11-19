# SSL Certificate Troubleshooting for Confident AI

## Issue
When running evaluations, you may see SSL certificate verification errors:
```
[SSLCertVerificationError: (1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)')]
```

## Impact
- **Evaluation still works**: All metrics are calculated and saved locally
- **Confident AI uploads fail**: Results won't be automatically uploaded to Confident AI dashboard
- **Non-blocking**: The evaluation completes successfully despite these errors

## Solutions

### Option 1: Install Python Certificates (Recommended for Production)

On macOS, install certificates using the `Install Certificates.command` script:

```bash
# Find your Python installation
python3 -c "import sys; print(sys.executable)"

# Navigate to the Python installation directory
cd /Applications/Python\ 3.12/
# Or wherever your Python is installed

# Run the certificate installer
./Install\ Certificates.command
```

Or manually:
```bash
# Install certifi package
pip install --upgrade certifi

# Set environment variable
export SSL_CERT_FILE=$(python3 -m certifi)
```

### Option 2: Disable SSL Verification (Development Only)

⚠️ **Warning**: Only use this for local development, never in production!

Add to your `.env` file:
```bash
# Disable SSL verification (development only)
PYTHONHTTPSVERIFY=0
```

Or set environment variable:
```bash
export PYTHONHTTPSVERIFY=0
```

### Option 3: Suppress Warnings (If Not Using Confident AI)

If you don't need Confident AI uploads, you can suppress the verbose logging:

```bash
export CONFIDENT_METRIC_LOGGING_VERBOSE=0
```

### Option 4: Use Local Evaluation Only

If you don't need Confident AI integration, simply don't set `CONFIDENT_API_KEY`. The evaluation will:
- ✅ Run all metrics locally
- ✅ Save results to `logs/eval_results.json`
- ✅ Display summary in terminal
- ❌ Skip Confident AI uploads (no errors)

## Verification

After applying a fix, run a quick test:
```bash
python evals/run_evaluation.py --squad-samples 1 --docvqa-samples 1 --fleurs-samples 1
```

Check for:
- ✅ No SSL errors in output
- ✅ Confident AI upload success messages (if using Option 1)
- ✅ Results saved to `logs/eval_results.json`

## Current Status

Your evaluation is working correctly! The SSL errors are cosmetic and don't affect:
- Metric calculations
- Local result storage
- Evaluation accuracy

You can safely ignore these warnings if you're only using local evaluation.

