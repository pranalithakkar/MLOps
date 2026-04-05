from locust import HttpUser, task, between

SAMPLE_INPUT = {
    'transaction_id': 'txn_a7f3c2e1-4b89-4d2a-89ac-12f345678901',
    'user_id': 'user_001',
    'description': 'KROGER #1247 SPRINGFIELD IL',
    'amount': 67.42,
    'currency': 'USD',
    'date': '2024-03-15',
    'account_id': 'acc_checking_001',
    'source': 'manual_entry'
}

SAMPLE_BATCH = {
    'transactions': [
        {**SAMPLE_INPUT, 'transaction_id': f'txn_batch_{i}'}
        for i in range(8)
    ]
}

class SingleUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(4)
    def predict_single(self):
        self.client.post('/predict', json=SAMPLE_INPUT)

    @task(1)
    def predict_batch(self):
        self.client.post('/predict_batch', json=SAMPLE_BATCH)

    @task(1)
    def health_check(self):
        self.client.get('/health')
