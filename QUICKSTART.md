# Quick Start Guide

Follow these steps to get the Multimodal Enterprise RAG system up and running:

## Step 1: Set Up Environment Variables

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API key:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

## Step 2: Start Docker Services

Start Neo4j and Qdrant using Docker Compose:

```bash
docker compose up -d
```

**Note:** If you get "command not found", try `docker-compose` (with hyphen) for older Docker versions, or use `docker compose` (with space) for Docker Compose V2.

This will start:
- **Neo4j** on http://localhost:7474 (web UI) and bolt://localhost:7687
- **Qdrant** on http://localhost:6333 (web UI) and localhost:6334 (gRPC)

Wait a few seconds for the services to be ready. You can check their status with:
```bash
docker compose ps
```

## Step 3: Install Python Dependencies

Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Step 4: Initialize Databases

Run the database initialization script:
```bash
python setup/init_databases.py
```

This will:
- Create necessary indexes and constraints in Neo4j
- Create the vector collection in Qdrant

You should see success messages for both databases.

## Step 5: Run the Streamlit UI

Start the web interface:
```bash
streamlit run ui/app.py
```

The UI will open in your browser at http://localhost:8501

## Step 6: Upload and Process Files

1. In the Streamlit UI, go to the **"File Upload"** page
2. Upload files (PDF, TXT, JPG, PNG, MP3)
3. Click **"Process Files"**
4. Wait for processing to complete

## Step 7: Query the System

1. Go to the **"Query"** page
2. Select a query type
3. Enter your question
4. Click **"Search"**
5. View the answer and sources

## Troubleshooting

### Docker Services Not Starting
- Check if ports 7474, 7687, 6333, 6334 are available
- Try: `docker compose down` then `docker compose up -d`
- If `docker compose` doesn't work, try `docker-compose` (older versions)

### Database Connection Errors
- Ensure Docker services are running: `docker compose ps`
- Check Neo4j web UI: http://localhost:7474 (login: neo4j/password)
- Check Qdrant web UI: http://localhost:6333

### OpenAI API Errors
- Verify your API key in `.env`
- Check your OpenAI account has credits
- Ensure you have access to GPT-4 models

### Import Errors
- Make sure you're in the project root directory
- Activate your virtual environment
- Reinstall dependencies: `pip install -r requirements.txt`

## Step 8: Run Evaluation (Optional)

To evaluate system performance:

1. **Ingest test data** (first time only):
```bash
python evals/run_evaluation.py
```

2. **Skip ingestion** (if data already ingested):
```bash
python evals/run_evaluation.py --skip-ingestion
```

3. **View results**: Check `logs/eval_results.json` for detailed metrics

### Confident AI Integration (Optional)

To enable Confident AI reporting, add to your `.env`:
```bash
CONFIDENT_AI_API_KEY=your_api_key
CONFIDENT_AI_PROJECT=your_project_name
CONFIDENT_AI_ENABLED=true
```

After running evaluations, you'll get a link to the Confident AI dashboard for hosted reports.

## Next Steps

- Explore the **Graph Explorer** to see knowledge graph relationships
- Run **Evaluation** to test system performance
- Check logs in the `logs/` directory for detailed information
- View evaluation reports on Confident AI (if enabled)

## Stopping the System

To stop Docker services:
```bash
docker compose down
```

To stop Streamlit:
- Press `Ctrl+C` in the terminal

