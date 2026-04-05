from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_name = 'distilbert-base-uncased'
save_path = './onnx_model'

print('Exporting model to ONNX format...')
model = ORTModelForSequenceClassification.from_pretrained(
    model_name, export=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f'Done. ONNX model saved to {save_path}')
