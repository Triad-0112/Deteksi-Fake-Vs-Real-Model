import torch
import torch.nn as nn
from torchsummary import summary
import argparse
import os
import glob
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import base64
from io import BytesIO, StringIO
from torchviz import make_dot
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torch.nn.utils.prune as prune
import sys

# Import your model and data loader functions
from models import ResNet50
from data_loader import get_dataloaders, data_transforms

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 5

def visualize_and_print_weights(model):
    print("\n" + "="*95)
    print("--- Inspect Weight Sparsity (All Relevant Layers) ---")
    print("="*95)

    visualizations = {}

    # Iterate through all modules in the model to find layers that can be pruned
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                weights = module.weight.detach().cpu()

                print(f"\nInspecting Layer: '{name}'")
                print(f"Weight Tensor Shape: {weights.shape}")

                print("Example weight values (top-left corner):")
                if weights.dim() == 4: # Layer Conv
                    print(weights[0, 0, :4, :4])
                elif weights.dim() == 2: # Layer Linear
                    print(weights[:5, :5])

                # Create heatmap visualization
                plt.figure(figsize=(10, 8))

                # Reshape tensor for 2D visualization
                if weights.dim() == 4: # (out, in, kH, kW)
                    w_2d = weights.view(weights.shape[0], -1).numpy()
                else:
                    w_2d = weights.numpy()

                sns.heatmap(w_2d, cmap='gray_r', cbar=False)
                plt.title(f"Weight Sparsity Map for '{name}'\n(White pixels are pruned weights)", fontsize=16)
                plt.xlabel("Input Connections")
                plt.ylabel("Output Neurons/Filters")

                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                # Use layer name as key to ensure uniqueness
                visualizations[name] = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()

                print(f"--> Visualization for '{name}' has been created.")

            except Exception as e:
                print(f"Cannot analyze layer '{name}': {e}")

    print("="*95 + "\n")
    return visualizations

def generate_pruned_summary_string(model):
    """Generate a formatted string from the model sparsity summary."""
    lines = []
    header = "--- Custom Model Pruning Summary (Sparsity Analysis) ---"
    lines.append("="*len(header))
    lines.append(header)
    lines.append("="*len(header))
    lines.append(f"{'Layer Name':<40} | {'Total Params':>15} | {'Non-Zero Params':>17} | {'Sparsity (%)':>15}")
    lines.append("-"*len(lines[-1]))
    
    total_params_overall, nonzero_params_overall = 0, 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total = module.weight.nelement()
            nonzero = torch.count_nonzero(module.weight).item()
            sparsity = 100. * (1 - (nonzero / total))
            total_params_overall += total
            nonzero_params_overall += nonzero
            lines.append(f"{name:<40} | {total:>15,} | {nonzero:>17,} | {sparsity:>14.2f}%")
            
    lines.append("-"*len(lines[-1]))
    if total_params_overall > 0:
        overall_sparsity = 100. * (1 - (nonzero_params_overall / total_params_overall))
        lines.append(f"{'TOTAL PRUNABLE':<40} | {total_params_overall:>15,} | {nonzero_params_overall:>17,} | {overall_sparsity:>14.2f}%")
    lines.append("="*len(header))
    return "\n".join(lines)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._save_feature_maps)
        self.target_layer.register_full_backward_hook(self._save_gradients)
    def _save_feature_maps(self, module, input, output): self.feature_maps = output.detach()
    def _save_gradients(self, module, grad_in, grad_out): self.gradients = grad_out[0].detach()
    def __call__(self, x):
        output = self.model(x)
        self.model.zero_grad()
        output.backward(torch.ones_like(output))
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = nn.functional.relu(cam); cam -= torch.min(cam); cam /= torch.max(cam)
        return cam.cpu().numpy().squeeze()

def generate_grad_cam_overlay(image_tensor, model, target_layer):
    original_image = image_tensor.permute(1, 2, 0).numpy()
    original_image = (original_image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    original_image = np.clip(original_image, 0, 1)
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam(image_tensor.unsqueeze(0).to(DEVICE))
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.uint8(255 * original_image), 0.6, heatmap, 0.4, 0)
    return np.uint8(255 * original_image), overlay

def get_sample_images():
    samples = {}
    transform = data_transforms['valid_test']
    sources = {'Dataset - Real': './dataset/test/real/*.*', 'Dataset - Fake': './dataset/test/fake/*.*', 'SynthBuster - Fake': './synthbuster/*/*.*'}
    for name, path in sources.items():
        try:
            files = glob.glob(path)
            if files: samples[name] = random.sample(files, min(NUM_SAMPLES, len(files)))
        except Exception as e: print(f"Could not load from {os.path.dirname(path)}: {e}")
    try:
        real_folders = glob.glob('./extracted_frames/Celeb-real/*/')
        fake_folders = glob.glob('./extracted_frames/Celeb-synthesis/*/')
        if real_folders: samples['Extracted Frames - Real'] = [glob.glob(f + '*.jpg')[0] for f in random.sample(real_folders, min(NUM_SAMPLES, len(real_folders)))]
        if fake_folders: samples['Extracted Frames - Fake'] = [glob.glob(f + '*.jpg')[0] for f in random.sample(fake_folders, min(NUM_SAMPLES, len(fake_folders)))]
    except Exception as e: print(f"Could not load from ./extracted_frames/: {e}")
    transformed_samples = {}
    for key, paths in samples.items():
        transformed_samples[key] = []
        for path in paths:
            try:
                img = np.array(Image.open(path).convert('RGB'))
                transformed_samples[key].append(transform(img))
            except Exception as e: print(f"Error loading sample {path}: {e}")
    return transformed_samples

def plot_metrics_to_base64(log_file):
    if not os.path.exists(log_file):
        print(f"Warning: Log file '{log_file}' not found.")
        return {}
    try:
        df = pd.read_html(log_file)[0]
    except Exception as e:
        print(f"Could not read or parse log file {log_file}: {e}")
        return {}
    plots = {}
    metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        plt.plot(df[df['Phase'] == 'train']['Epoch'], df[df['Phase'] == 'train'][metric], 'b-o', label=f'Training {metric}')
        plt.plot(df[df['Phase'] == 'valid']['Epoch'], df[df['Phase'] == 'valid'][metric], 'r-o', label=f'Validation {metric}')
        plt.title(f'Training vs Validation {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plots[metric] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    return plots

def image_to_base64(img_array):
    img = Image.fromarray(img_array)
    buf = BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_architecture_graph_base64(model):
    try:
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        graph = make_dot(model(dummy_input), params=dict(model.named_parameters()))
        temp_filename = "temp_arch_graph"
        graph.render(temp_filename, format="png", cleanup=True)
        with open(f"{temp_filename}.png", "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Could not generate torchviz graph: {e}. Is graphviz installed?")
        return None

def evaluate_model_on_test_set(model):
    dataloaders, _ = get_dataloaders()
    test_loader = dataloaders.get('test')
    if not test_loader:
        print("Warning: Test data loader not found.")
        return None, None
    all_labels, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test Set Evaluation"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())
    return np.array(all_labels), np.array(all_probs)

def generate_evaluation_plots(y_true, y_pred_probs):
    if y_true is None or y_pred_probs is None or len(y_true) == 0:
        return None, None, None
    plots = {}
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    auroc_score = auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plots['ROC Curve'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    y_pred_class = (y_pred_probs > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plots['Confusion Matrix'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return plots, auroc_score

def generate_html_report(model_path, model_summary_str, pruned_summary_str, arch_graph_b64, plots, eval_plots, auroc_score, grad_cam_results, weight_visualizations):
    html = f"""
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Training Report</title><script src="https://cdn.tailwindcss.com"></script></head>
    <body class="bg-gray-100 font-sans"><div class="container mx-auto p-8"><div class="bg-white p-8 rounded-lg shadow-lg">
    <h1 class="text-4xl font-bold mb-4 text-gray-800">Model Training Report</h1>
    <p class="text-gray-600 mb-6">Generated for model: <code class="bg-gray-200 p-1 rounded">{model_path}</code></p>
    <div class="mb-12"><h2 class="text-3xl font-semibold mb-4 text-gray-700 border-b-2 pb-2">Model Architecture</h2>
    <h3 class="text-xl font-semibold mb-2">Text Summary</h3><div class="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mb-6"><pre>{model_summary_str}</pre></div>"""
    
    if pruned_summary_str:
        html += f"""<h3 class="text-xl font-semibold mb-2">Sparsity Analysis</h3>
        <div class="bg-gray-800 text-white p-4 rounded-md overflow-x-auto mb-6"><pre>{pruned_summary_str}</pre></div>"""
    
    if arch_graph_b64:
        html += f"""<h3 class="text-xl font-semibold mb-2">Graphical Visualization</h3><div class="bg-white p-4 rounded-lg shadow">
        <img src="data:image/png;base64,{arch_graph_b64}" alt="Model Architecture Graph" class="w-full h-auto rounded"></div>"""
    html += """</div>"""

    if weight_visualizations:
        html += """<div class="mb-12"><h2 class="text-3xl font-semibold mb-4 text-gray-700 border-b-2 pb-2">Visualisasi Sparsity Bobot</h2>
        <p class="text-gray-600 mb-6">Heatmap from weight matrix for sample layer. White pixel represents pruned (zeroed) weights.</p>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">"""
        for name, b64_img in weight_visualizations.items():
            html += f"""<div class="bg-white p-4 rounded-lg shadow"><h3 class="text-xl font-semibold mb-2 text-center">{name}</h3>
            <img src="data:image/png;base64,{b64_img}" alt="Weight Sparsity for {name}" class="w-full h-auto rounded"></div>"""
        html += """</div></div>"""

    if plots:
        html += """<div class="mb-12"><h2 class="text-3xl font-semibold mb-4 text-gray-700 border-b-2 pb-2">Training & Validation Metrics</h2><div class="grid grid-cols-1 md:grid-cols-2 gap-8">"""
        for metric, b64_img in plots.items():
            html += f"""<div class="bg-white p-4 rounded-lg shadow"><h3 class="text-xl font-semibold mb-2 text-center">{metric} Over Epochs</h3><img src="data:image/png;base64,{b64_img}" alt="{metric} Plot" class="w-full h-auto rounded"></div>"""
        html += """</div></div>"""
    if eval_plots:
        auroc_text = f"{auroc_score:.4f}" if auroc_score is not None else "N/A"
        html += f"""<div class="mb-12"><h2 class="text-3xl font-semibold mb-4 text-gray-700 border-b-2 pb-2">Final Test Set Evaluation</h2><p class="text-gray-600 mb-4"><b>Area Under ROC Curve (AUROC):</b> {auroc_text}</p><div class="grid grid-cols-1 md:grid-cols-2 gap-8">"""
        for name, b64_img in eval_plots.items():
            html += f"""<div class="bg-white p-4 rounded-lg shadow"><h3 class="text-xl font-semibold mb-2 text-center">{name}</h3><img src="data:image/png;base64,{b64_img}" alt="{name} Plot" class="w-full h-auto rounded"></div>"""
        html += """</div></div>"""
    if grad_cam_results:
        html += """<div><h2 class="text-3xl font-semibold mb-4 text-gray-700 border-b-2 pb-2">Grad-CAM Analysis</h2><p class="text-gray-600 mb-6">Visualizing model's attention focus on sample images.</p>"""
        for category, results in grad_cam_results.items():
            html += f"""<div class="mb-8"><h3 class="text-2xl font-semibold mb-4 text-gray-700">{category}</h3><div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">"""
            for original_b64, overlay_b64 in results:
                html += f"""<div class="bg-white p-2 rounded shadow"><p class="text-center text-sm font-medium mb-1">Original</p><img src="data:image/png;base64,{original_b64}" class="w-full h-auto rounded mb-2"><p class="text-center text-sm font-medium mb-1">Grad-CAM</p><img src="data:image/png;base64,{overlay_b64}" class="w-full h-auto rounded"></div>"""
            html += """</div></div>"""
        html += """</div>"""
    html += """</div></div></body></html>"""
    return html

def main():
    parser = argparse.ArgumentParser(description="Generate a training report for a ResNet model.")
    parser.add_argument("model_path", type=str, help="Path to the trained model .pth file.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    print("Memuat model...")
    model = ResNet50(num_classes=1)
    
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    pruned_summary_str = generate_pruned_summary_string(model)
    print(pruned_summary_str)
    weight_visualizations = visualize_and_print_weights(model)

    print("Create standard model summary for HTML report...")
    summary_buffer = StringIO()
    old_stdout = sys.stdout
    sys.stdout = summary_buffer
    summary(model, (3, 224, 224))
    sys.stdout = old_stdout
    model_summary_str = summary_buffer.getvalue()

    print("Create model architecture graph...")
    arch_graph_b64 = generate_architecture_graph_base64(model)

    model_name_from_path = os.path.splitext(os.path.basename(args.model_path))[0]
    log_name_parts = model_name_from_path.split('_')
    log_model_name = f"{log_name_parts[0]}_{log_name_parts[1]}" if 'prune' in model_name_from_path else log_name_parts[0]
    log_filename = f"training_log_{log_model_name}.html"
    log_file_path = os.path.join("Report", "Model_Train", log_filename)

    print(f"Create metric plots from: {log_file_path}")
    plots = plot_metrics_to_base64(log_file_path)

    y_true, y_probs = evaluate_model_on_test_set(model)
    eval_plots, auroc_score = generate_evaluation_plots(y_true, y_probs)

    print("Create Grad-CAM visualizations...")
    sample_images = get_sample_images()
    grad_cam_results = {}
    target_layer = model.layer4[-1] 
    for category, images in sample_images.items():
        grad_cam_results[category] = []
        for img_tensor in images:
            original_img_np, overlay_np = generate_grad_cam_overlay(img_tensor, model, target_layer)
            grad_cam_results[category].append((image_to_base64(original_img_np), image_to_base64(overlay_np)))

    print("Create HTML report...")
    report_html = generate_html_report(args.model_path, model_summary_str, pruned_summary_str, arch_graph_b64, plots, eval_plots, auroc_score, grad_cam_results, weight_visualizations)
    
    output_dir = os.path.join("Report", "Model_Summary")
    os.makedirs(output_dir, exist_ok=True)
    report_model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    report_filename = f"summary_{report_model_name}.html"
    full_report_path = os.path.join(output_dir, report_filename)

    with open(full_report_path, "w", encoding='utf-8') as f:
        f.write(report_html)
    print(f"\nReport successfully created. Open '{full_report_path}' in your browser.")

if __name__ == '__main__':
    main()

