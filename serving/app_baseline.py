from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from datetime import datetime, timezone
import time

app = FastAPI()

# Model loads ONCE when the server starts, not on every request
print('Loading model...')
classifier = pipeline(
    'text-classification',
    model='distilbert-base-uncased',
    top_k=None
)
print('Model loaded.')

# This defines what a valid incoming request looks like
# All fields match input_sample_categorization.json exactly
class Transaction(BaseModel):
    transaction_id: str
    user_id: str = ''
    description: str
    amount: float = 0.0
    currency: str = 'USD'
    date: str = ''
    account_id: str = ''
    source: str = 'manual_entry'

# Health check endpoint — course staff uses this to verify your server is alive
@app.get('/health')
def health():
    return {'status': 'ok'}

# Main prediction endpoint
@app.post('/predict')
def predict(tx: Transaction):
    start = time.time()
    results = classifier(tx.description)
    latency_ms = round((time.time() - start) * 1000, 2)

    # Sort all categories by score, take top 3
    sorted_results = sorted(
        results[0], key=lambda x: x['score'], reverse=True
    )[:3]
    top = sorted_results[0]
    confidence = round(top['score'], 4)

    predictions = [
        {'category': r['label'], 'probability': round(r['score'], 4)}
        for r in sorted_results
    ]

    # Response matches output_sample_categorization.json exactly
    return {
        'transaction_id': tx.transaction_id,
        'model': 'distilbert-categorization',
        'model_version': '1.2.0',
        'predictions': predictions,
        'top_category': top['label'],
        'confidence': confidence,
        'abstain': confidence < 0.7,
        'inference_latency_ms': latency_ms,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }

# Batch endpoint — accepts multiple transactions in one request
class BatchTransaction(BaseModel):
    transactions: list[Transaction]

@app.post('/predict_batch')
def predict_batch(batch: BatchTransaction):
    start = time.time()
    descriptions = [tx.description for tx in batch.transactions]
    all_results = classifier(descriptions, batch_size=8)
    total_latency_ms = round((time.time() - start) * 1000, 2)

    output = []
    for tx, results in zip(batch.transactions, all_results):
        sorted_r = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
        top = sorted_r[0]
        confidence = round(top['score'], 4)
        output.append({
            'transaction_id': tx.transaction_id,
            'top_category': top['label'],
            'confidence': confidence,
            'abstain': confidence < 0.7,
            'predictions': [
                {'category': r['label'], 'probability': round(r['score'], 4)}
                for r in sorted_r
            ]
        })

    return {
        'results': output,
        'count': len(output),
        'total_latency_ms': total_latency_ms,
        'model_version': '1.2.0',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
