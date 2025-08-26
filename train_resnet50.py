# train_resnet50.py

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

from models import ResNet50
from data_loader import get_dataloaders

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
MODEL_SAVE_PATH = './models/'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

PATIENCE = 3 # Early Stopping
LOG_FILE = 'training_log.html'
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'resnet50_best_model.pth')
LATEST_CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, 'resnet50_latest_checkpoint.pth')

class HtmlLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("<html><head><title>Training Log</title></head><body>")
                f.write("<h1>Training Log</h1>")
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
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0
        self.path = path

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_max:.4f} --> {val_acc:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_acc_max = val_acc

def train_model():
    logger = HtmlLogger(LOG_FILE)
    
    # --- Load Data ---
    dataloaders, dataset_sizes = get_dataloaders()

    model = ResNet50(num_classes=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    scaler = GradScaler()
    
    start_epoch = 0
    best_acc = 0.0
    if os.path.exists(LATEST_CHECKPOINT_PATH):
        print("Resuming training from latest checkpoint...")
        logger.log_message("Resuming training from latest checkpoint.")
        checkpoint = torch.load(LATEST_CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resuming from Epoch {start_epoch}")
    
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=BEST_MODEL_PATH)
    early_stopping.val_acc_max = best_acc

    since = time.time()
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f'Epoch {epoch}/{NUM_EPOCHS - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            
            all_labels = []
            all_preds = []

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
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
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
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, LATEST_CHECKPOINT_PATH)
        
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            logger.log_message("Early stopping triggered.")
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    logger.log_message(f'Training complete. Best val Acc: {best_acc:4f}')
    logger.close()
    
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    return model

if __name__ == '__main__':
    trained_model = train_model()
    print("Training finished.")
