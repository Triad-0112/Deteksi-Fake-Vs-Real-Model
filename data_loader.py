# data_loader.py

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import glob
import random
from sklearn.model_selection import train_test_split
import warnings

# --- Configuration ---
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4

warnings.filterwarnings('ignore', category=FutureWarning)

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid_test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class FinalDataset(Dataset):
    def __init__(self, path_list, labels, transform=None):
        self.path_list = path_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        media_path = self.path_list[idx]
        label = self.labels[idx]
        try:
            if os.path.isdir(media_path):
                frame_paths = glob.glob(os.path.join(media_path, '*.jpg'))
                if not frame_paths: raise IOError(f"No frames found in {media_path}")
                
                chosen_path = random.choice(frame_paths)
                image = np.array(Image.open(chosen_path).convert('RGB'))
                image_tensor = self.transform(image)
            
            else:
                image = np.array(Image.open(media_path).convert('RGB'))
                image_tensor = self.transform(image)
                
        except Exception as e:
            print(f"Error loading {media_path}: {e}. Returning a placeholder.")
            image_tensor = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))
            label = 0 
            
        return image_tensor, torch.tensor(label, dtype=torch.float32)

def get_dataloaders():
    """
    This function consolidates all data loading logic and returns the dataloaders.
    """
    # --- DATASET PATHS ---
    IMAGE_DATA_DIR = './dataset/'
    SYNTH_DATA_DIR = './synthbuster'
    EXTRACTED_FRAMES_DIR = './extracted_frames/'

    def get_frame_folders(data_dir, class_map):
        folder_list, labels = [], []
        for cls, label in class_map.items():
            class_path = os.path.join(data_dir, cls)
            if os.path.isdir(class_path):
                video_folders = [f.path for f in os.scandir(class_path) if f.is_dir()]
                folder_list.extend(video_folders)
                labels.extend([label] * len(video_folders))
        return folder_list, labels

    def get_image_files(data_dir, class_map):
        file_list, labels = [], []
        for cls, label in class_map.items():
            class_path = os.path.join(data_dir, cls)
            if os.path.isdir(class_path):
                for fname in glob.glob(os.path.join(class_path, "**", "*.*"), recursive=True):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_list.append(fname)
                        labels.append(label)
        return file_list, labels

    image_class_map = {'real': 1, 'fake': 0}
    video_frame_class_map = {'Celeb-real': 1, 'Celeb-synthesis': 0}

    datasets_paths = {'train': [], 'valid': [], 'test': []}
    datasets_labels = {'train': [], 'valid': [], 'test': []}

    print("Loading original image dataset...")
    for split in ['train', 'valid', 'test']:
        img_files, img_labels = get_image_files(os.path.join(IMAGE_DATA_DIR, split), image_class_map)
        datasets_paths[split].extend(img_files)
        datasets_labels[split].extend(img_labels)

    print("Loading extracted frames dataset...")
    all_frame_folders, all_frame_labels = get_frame_folders(EXTRACTED_FRAMES_DIR, video_frame_class_map)
    vid_train_folders, vid_valid_folders, vid_test_folders = [], [], []
    vid_train_labels, vid_valid_labels, vid_test_labels = [], [], []

    if all_frame_folders:
        vid_train_folders, vid_temp_folders, vid_train_labels, vid_temp_labels = train_test_split(
            all_frame_folders, all_frame_labels, test_size=0.2, random_state=42, stratify=all_frame_labels)
        vid_valid_folders, vid_test_folders, vid_valid_labels, vid_test_labels = train_test_split(
            vid_temp_folders, vid_temp_labels, test_size=0.5, random_state=42, stratify=vid_temp_labels)
    else:
        print("Warning: No data found in 'extracted_frames'. Skipping this dataset.")

    print("Loading SynthBuster dataset...")
    all_synth_files = []
    if os.path.exists(SYNTH_DATA_DIR):
        synth_models = [d.name for d in os.scandir(SYNTH_DATA_DIR) if d.is_dir()]
        for model_dir in synth_models:
            synth_map = {model_dir: 0}
            synth_files, _ = get_image_files(SYNTH_DATA_DIR, synth_map)
            all_synth_files.extend(synth_files)
    all_synth_labels = [0] * len(all_synth_files)
    
    synth_train_files, synth_valid_files, synth_test_files = [], [], []
    synth_train_labels, synth_valid_labels, synth_test_labels = [], [], []

    if all_synth_files:
        synth_train_files, synth_temp_files, synth_train_labels, synth_temp_labels = train_test_split(
            all_synth_files, all_synth_labels, test_size=0.2, random_state=42)
        synth_valid_files, synth_test_files, synth_valid_labels, synth_test_labels = train_test_split(
            synth_temp_files, synth_temp_labels, test_size=0.5, random_state=42)
    else:
        print("Warning: No data found in 'synthbuster'. Skipping this dataset.")

    print("Combining all datasets...")
    datasets_paths['train'].extend(vid_train_folders)
    datasets_paths['train'].extend(synth_train_files)
    datasets_labels['train'].extend(vid_train_labels)
    datasets_labels['train'].extend(synth_train_labels)

    datasets_paths['valid'].extend(vid_valid_folders)
    datasets_paths['valid'].extend(synth_valid_files)
    datasets_labels['valid'].extend(vid_valid_labels)
    datasets_labels['valid'].extend(synth_valid_labels)

    datasets_paths['test'].extend(vid_test_folders)
    datasets_paths['test'].extend(synth_test_files)
    datasets_labels['test'].extend(vid_test_labels)
    datasets_labels['test'].extend(synth_test_labels)

    final_datasets = {
        'train': FinalDataset(datasets_paths['train'], datasets_labels['train'], data_transforms['train']),
        'valid': FinalDataset(datasets_paths['valid'], datasets_labels['valid'], data_transforms['valid_test']),
        'test': FinalDataset(datasets_paths['test'], datasets_labels['test'], data_transforms['valid_test'])
    }

    dataloaders = {
        x: DataLoader(final_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=NUM_WORKERS, pin_memory=True) 
        for x in ['train', 'valid', 'test']
    }
    dataset_sizes = {x: len(final_datasets[x]) for x in ['train', 'valid', 'test']}
    
    print("\nFinal combined dataset sizes:")
    for split, size in dataset_sizes.items():
        print(f"{split.capitalize()} : {size}")
        
    return dataloaders, dataset_sizes
