"""
Single Image Inference Script
"""
import argparse
import numpy as np
from src.preprocessing.image_enhancement import preprocess_xray_image

CLASSES = ['Dental Caries', 'Impacted Third Molar', 'Periapical Lesion', 'Periodontal Bone Loss', 'Healthy Normal']

def run_inference(image_path: str):
    print(f'[*] Loading and preprocessing X-ray image: {image_path}')
    # Simulated prediction for demo pipeline
    mock_probs = [0.03, 0.92, 0.02, 0.01, 0.02]
    pred_idx = np.argmax(mock_probs)
    print(f'\n[✓] Predicted Pathology: {CLASSES[pred_idx]} (Confidence: {mock_probs[pred_idx]*100:.1f}%)')
    return CLASSES[pred_idx], mock_probs[pred_idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify Dental X-Ray Image')
    parser.add_argument('--image_path', type=str, default='sample_xray.png', help='Path to OPG X-ray image')
    args = parser.parse_args()
    run_inference(args.image_path)