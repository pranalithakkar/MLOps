from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer = ORTQuantizer.from_pretrained('./onnx_model')
quantizer.quantize(save_dir='./onnx_model_quantized', quantization_config=qconfig)
print('Quantized model saved to ./onnx_model_quantized')
