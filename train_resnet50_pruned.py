import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
import os
import copy
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.utils.prune as prune
import argparse

from models import ResNet50
from data_loader import get_dataloaders

def train_pruned_model(prune_percentage):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 100
    MODEL_SAVE_PATH = './models/'
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    prune_str = f"prune{int(prune_percentage * 100)}pct"
    
    PATIENCE = 3
    LOG_FILE = os.path.join("Report", "Model_Train", f'training_log_resnet50_{prune_str}.html')
    BEST_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, f'resnet50_{prune_str}_best_model.pth')
    LATEST_CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, f'resnet50_{prune_str}_latest_checkpoint.pth')
    ORIGINAL_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'resnet50_best_model.pth')

    class HtmlLogger:
        def __init__(self, log_file):
            self.log_file = log_file
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            if not os.path.exists(self.log_file):
                with open(self.log_file, "w") as f:
                    f.write("<html><head><title>Training Log</title></head><body><h1>Training Log</h1>")
                    f.write("<table border='1'><tr><th>Timestamp</th><th>Epoch</th><th>Phase</th><th>Loss</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>")
        def log(self, epoch, phase, loss, acc, precision, recall, f1):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a") as f:
                f.write(f"<tr><td>{timestamp}</td><td>{epoch}</td><td>{phase}</td><td>{loss:.4f}</td><td>{acc:.4f}</td><td>{precision:.4f}</td><td>{recall:.4f}</td><td>{f1:.4f}</td></tr>")
        def log_message(self, message):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a") as f:
                f.write(f"<p><b>{timestamp}:</b> {message}</p>")
        def close(self):
            with open(self.log_file, "a") as f:
                f.write("</table></body></html>")

    class EarlyStopping:
        def __init__(self, patience=3, verbose=True, path='checkpoint.pth'):
            self.patience, self.verbose, self.path = patience, verbose, path
            self.counter, self.best_score, self.early_stop, self.val_acc_max = 0, None, False, 0
        
        def __call__(self, val_acc, model):
            score = val_acc
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.save_checkpoint(val_acc, model)
                self.counter = 0
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience: self.early_stop = True
        
        def save_checkpoint(self, val_acc, model):
            if self.verbose: print(f'Validation accuracy increased ({self.val_acc_max:.4f} --> {val_acc:.4f}). Saving model ...')
            torch.save(model.state_dict(), self.path)
            self.val_acc_max = val_acc


    logger = HtmlLogger(LOG_FILE)
    
    dataloaders, dataset_sizes = get_dataloaders()

    model = ResNet50(num_classes=1).to(DEVICE)
    if not os.path.exists(ORIGINAL_MODEL_PATH):
        print(f"Error: Original model not found at {ORIGINAL_MODEL_PATH}. Please train the base model first.")
        return
        
    print(f"Loading pre-trained model from {ORIGINAL_MODEL_PATH}")
    model.load_state_dict(torch.load(ORIGINAL_MODEL_PATH))

    print(f"Applying {prune_percentage*100:.0f}% global unstructured pruning...")
    logger.log_message(f"Applying {prune_percentage*100:.0f}% global unstructured pruning...")
    
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_percentage,
    )

    total_params, zero_params = 0, 0
    for module, name in parameters_to_prune:
        total_params += module.weight.nelement()
        zero_params += torch.sum(module.weight == 0)
    
    sparsity = 100. * float(zero_params) / float(total_params)
    print(f"Model sparsity: {sparsity:.2f}%")
    logger.log_message(f"Model sparsity after pruning: {sparsity:.2f}%")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()
    
    start_epoch, best_acc = 0, 0.0
    if os.path.exists(LATEST_CHECKPOINT_PATH):
        print(f"Resuming training from latest checkpoint: {LATEST_CHECKPOINT_PATH}")
        logger.log_message("Resuming training from latest checkpoint.")
        checkpoint = torch.load(LATEST_CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resuming from Epoch {start_epoch}")
    
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=BEST_MODEL_PATH)
    
    since = time.time()
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f'Epoch {epoch}/{NUM_EPOCHS - 1}'); print('-' * 10)
        for phase in ['train', 'valid']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            all_labels, all_preds = [], []

            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch}")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    preds = (outputs > 0).float()
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy()); all_preds.extend(preds.cpu().numpy())
                progress_bar.set_postfix(loss=loss.item())

            if phase == 'train': scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}')
            logger.log(epoch, phase, epoch_loss, epoch_acc, precision, recall, f1)
            if phase == 'valid':
                best_acc = max(best_acc, epoch_acc)
                early_stopping(epoch_acc, model)
        
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, LATEST_CHECKPOINT_PATH)
        
        if early_stopping.early_stop:
            print("Early stopping triggered."); logger.log_message("Early stopping triggered.")
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    logger.log_message(f'Training complete. Best val Acc: {best_acc:4f}')
    
    print("Finalizing the best pruned model...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    for module, name in parameters_to_prune:
        if prune.is_pruned(module):
            prune.remove(module, name)
    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print(f"Best pruned model saved permanently to {BEST_MODEL_PATH}")
    
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune a pruned ResNet-50 model.")
    parser.add_argument("--prune_pct", type=float, required=True, help="The percentage of weights to prune (e.g., 0.1 for 10%).")
    args = parser.parse_args()
    
    if not 0 < args.prune_pct < 1:
        raise ValueError("Pruning percentage must be between 0.0 and 1.0.")
        
    train_pruned_model(args.prune_pct)
    print("Pruning and fine-tuning finished.")
