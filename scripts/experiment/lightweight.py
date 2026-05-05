import os
import sys
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from core.models.model_definitions import BERTTextCNN

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'experiments', 'lightweight')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def layer_pruning(model, prune_ratio=0.5):
    logger.info(f"  Layer pruning: removing {prune_ratio*100:.0f}% of BERT layers...")
    total_layers = len(model.bert.encoder.layer)
    keep_layers = int(total_layers * (1 - prune_ratio))
    keep_indices = sorted(np.linspace(0, total_layers - 1, keep_layers, dtype=int).tolist())

    logger.info(f"    Original: {total_layers} layers, After pruning: {keep_layers} layers")
    logger.info(f"    Keeping layers: {keep_indices}")

    pruned_layers = nn.ModuleList([model.bert.encoder.layer[i] for i in keep_indices])
    model.bert.encoder.layer = pruned_layers

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"    After pruning: total={total_params:,}, trainable={trainable_params:,}")

    return model


def fp16_quantization(model):
    logger.info("  FP16 quantization...")
    model = model.half()
    total_params = sum(p.numel() for p in model.parameters())
    size_fp32 = total_params * 4 / (1024 * 1024)
    size_fp16 = total_params * 2 / (1024 * 1024)
    logger.info(f"    FP32 size: {size_fp32:.1f} MB")
    logger.info(f"    FP16 size: {size_fp16:.1f} MB")
    logger.info(f"    Compression ratio: {size_fp32/size_fp16:.1f}x")
    return model


def onnx_export(model, tokenizer, sample_text="【银行】您的账户存在安全风险，请立即验证", output_path=None):
    logger.info("  ONNX export...")
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'bert_textcnn.onnx')

    device = next(model.parameters()).device
    model.eval()
    model = model.float()

    enc = tokenizer(sample_text, max_length=128, padding='max_length',
                    truncation=True, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    class ONNXWrapper(nn.Module):
        def __init__(self, bert_textcnn):
            super().__init__()
            self.model = bert_textcnn

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, attention_mask)

    wrapper = ONNXWrapper(model)

    try:
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        onnx_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"    ONNX model saved: {output_path} ({onnx_size:.1f} MB)")

        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(output_path)
            ort_inputs = {
                'input_ids': input_ids.cpu().numpy(),
                'attention_mask': attention_mask.cpu().numpy(),
            }
            ort_output = sess.run(None, ort_inputs)
            logger.info(f"    ONNX Runtime verification: output shape={ort_output[0].shape}")
        except ImportError:
            logger.warning("    onnxruntime not installed, skipping ONNX Runtime verification")

        return output_path
    except Exception as e:
        logger.error(f"    ONNX export failed: {e}")
        return None


def measure_inference_speed(model, tokenizer, texts, device, batch_size=32):
    model.eval()
    model = model.float()

    encodings = tokenizer(texts, max_length=128, padding='max_length',
                          truncation=True, return_tensors='pt')

    dataset = torch.utils.data.TensorDataset(
        encodings['input_ids'], encodings['attention_mask']
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for _ in range(3):
            for batch in loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                model(input_ids, attention_mask)

    start = time.time()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            model(input_ids, attention_mask)
    elapsed = time.time() - start

    speed = len(texts) / elapsed
    logger.info(f"    Inference speed: {speed:.1f} samples/sec ({len(texts)} samples in {elapsed:.2f}s)")
    return speed


def run_lightweight_pipeline():
    logger.info("=" * 60)
    logger.info("  BERT-TextCNN Lightweight Deployment Pipeline")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

    best_model_path = os.path.join(MODEL_DIR, 'bert_textcnn_best.pth')
    if not os.path.exists(best_model_path):
        logger.error(f"Model not found: {best_model_path}")
        return

    logger.info("\n[Step 1] Loading original model...")
    bert = AutoModel.from_pretrained('bert-base-chinese')
    original_model = BERTTextCNN(bert).to(device)
    state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
    original_model.load_state_dict(state_dict)
    original_model.eval()

    total_params = sum(p.numel() for p in original_model.parameters())
    trainable_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
    original_size = total_params * 4 / (1024 * 1024)
    logger.info(f"  Original: total={total_params:,}, trainable={trainable_params:,}, size={original_size:.1f} MB")

    logger.info("\n[Step 2] Layer pruning (50%)...")
    pruned_model = layer_pruning(original_model, prune_ratio=0.5)
    pruned_model.eval()

    logger.info("\n[Step 3] FP16 quantization...")
    fp16_model = fp16_quantization(pruned_model)
    fp16_model.eval()

    logger.info("\n[Step 4] ONNX export...")
    onnx_path = onnx_export(pruned_model, tokenizer)

    logger.info("\n[Step 5] Measuring inference speed...")
    test_texts = ["【银行】您的账户存在安全风险，请立即验证身份"] * 200
    original_speed = measure_inference_speed(original_model, tokenizer, test_texts, device)

    pruned_float = pruned_model.float()
    pruned_speed = measure_inference_speed(pruned_float, tokenizer, test_texts, device)

    logger.info("\n[Step 6] Saving lightweight model...")
    pruned_path = os.path.join(OUTPUT_DIR, 'bert_textcnn_pruned.pth')
    torch.save(pruned_model.state_dict(), pruned_path)
    pruned_file_size = os.path.getsize(pruned_path) / (1024 * 1024)
    logger.info(f"  Pruned model saved: {pruned_path} ({pruned_file_size:.1f} MB)")

    report = {
        'original': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': round(original_size, 1),
            'inference_speed': round(original_speed, 1),
            'bert_layers': 12,
        },
        'pruned_fp16': {
            'total_params': sum(p.numel() for p in pruned_model.parameters()),
            'model_size_mb': round(pruned_file_size, 1),
            'inference_speed': round(pruned_speed, 1),
            'bert_layers': 6,
            'prune_ratio': 0.5,
            'precision': 'FP16',
        },
        'onnx': {
            'exported': onnx_path is not None,
            'model_size_mb': round(os.path.getsize(onnx_path) / (1024 * 1024), 1) if onnx_path else None,
            'path': onnx_path,
        },
        'compression_ratio': round(original_size / pruned_file_size, 2) if pruned_file_size > 0 else None,
        'speed_improvement': round(pruned_speed / original_speed, 2) if original_speed > 0 else None,
    }

    report_path = os.path.join(OUTPUT_DIR, 'lightweight_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"\n  Report saved: {report_path}")

    logger.info("\n" + "=" * 60)
    logger.info("  LIGHTWEIGHT DEPLOYMENT COMPLETE")
    logger.info(f"  Original: {original_size:.1f} MB, {original_speed:.1f} samples/sec")
    logger.info(f"  Pruned+FP16: {pruned_file_size:.1f} MB, {pruned_speed:.1f} samples/sec")
    logger.info(f"  Compression: {report['compression_ratio']}x, Speed: {report['speed_improvement']}x")
    logger.info("=" * 60)

    return report


if __name__ == '__main__':
    run_lightweight_pipeline()
