from fastapi import FastAPI
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline
from datetime import datetime, timezone
import time

app = FastAPI()

print('Loading ONNX model...')
model = ORTModelForSequenceClassification.from_pretrained('./onnx_model')
tokenizer = AutoTokenizer.from_pretrained('./onnx_model')
classifier = pipeline(
    'text-classification',
    model=model,
    tokenizer=tokenizer,
    top_k=None
)
print('ONNX model loaded.')

class Transaction(BaseModel):
    transaction_id: str
    user_id: str = ''
    description: str
    amount: float = 0.0
    currency: str = 'USD'
    date: str = ''
    account_id: str = ''
    source: str = 'manual_entry'

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict(tx: Transaction):
    start = time.time()
    results = classifier(tx.description)
    latency_ms = round((time.time() - start) * 1000, 2)

    sorted_results = sorted(
        results[0], key=lambda x: x['score'], reverse=True
    )[:3]
    top = sorted_results[0]
    confidence = round(top['score'], 4)

    return {
        'transaction_id': tx.transaction_id,
        'model': 'distilbert-categorization',
        'model_version': '1.2.0-onnx',
        'predictions': [
            {'category': r['label'], 'probability': round(r['score'], 4)}
            for r in sorted_results
        ],
        'top_category': top['label'],
        'confidence': confidence,
        'abstain': confidence < 0.7,
        'inference_latency_ms': latency_ms,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }

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
        'model_version': '1.2.0-onnx',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
