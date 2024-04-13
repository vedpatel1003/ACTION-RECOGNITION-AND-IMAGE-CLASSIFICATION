import cv2
import os
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import PIL

dataset_path = "/home/ved/projects/action_recognition/pytorch_learning/UCF101/UCF-101"
output_dir = os.path.join(dataset_path, "extracted_frames")
os.makedirs(output_dir, exist_ok=True) 

def extract_frames(video_paths, output_filename_prefix, frame_interval=1):
    video_path = random.choice(video_paths)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % frame_interval == 0:
            filename = f"{output_filename_prefix}_{count}.jpg"
            cv2.imwrite(filename, image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()

video_paths = ["/home/ved/projects/action_recognition/pytorch_learning/UCF101/UCF-101"]
output_filename_prefix = "output_frame"
frame_interval = 10
extract_frames(video_paths, output_filename_prefix, frame_interval)

for root, _, video_files in os.walk(dataset_path):
    for video_file in video_files:
        if video_file.endswith(".avi") or video_file.endswith(".mp4"):
            video_path = os.path.join(root, video_file)
            class_name = os.path.basename(os.path.dirname(video_path))
            output_filename_prefix = f"{class_name}_{video_file[:-4]}"
            extract_frames(video_path, output_filename_prefix)

print("Frame extraction completed!")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

class UCF101Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_labels = self.load_class_labels()
        self.video_paths = self.load_video_paths()
        self.frame_counts = self.load_frame_counts()

    def load_class_labels(self):
        class_file_path = os.path.join(self.data_dir, 'classInd.txt')
        with open(class_file_path, 'r') as f:
            class_labels = [line.strip() for line in f]
        return class_labels
    
    def load_video_paths(self):
        video_file_path = os.path.join(self.data_dir, 'video_list.txt')
        with open(video_file_path, 'r') as f:
            video_paths = [line.strip() for line in f]
        return video_paths
    
    def load_frame_counts(self):
        frame_count_file_path = os.path.join(self.data_dir, 'frame_counts.txt')
        with open(frame_count_file_path, 'r') as f:
            frame_counts = [int(line.strip()) for line in f]
        return frame_counts
    
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_dir, self.video_paths[idx])
        frame_count = self.frame_counts[idx]
        frames = self.load_frames(video_path, frame_count)
        label = self.class_labels.index(self.video_paths[idx].split('/')[0])
        if self.transform:
            frames = self.transform(frames)
        return frames, label
    
    def load_frames(self, video_path, frame_count):
        frames = []
        for i in range(frame_count):
            frame_path = os.path.join(video_path, f'frame_{i}.jpg')
            frame = Image.open(frame_path)
            frame = frame.convert('RGB')
            frames.append(frame)
        return frames
        
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = UCF101Dataset(dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for i, (frames, label) in enumerate(dataloader):
    print(f"Batch {i+1}: Frames shape = {frames.shape}, Label = {label}")
    break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10
for epoch in range(num_epochs):
    for i, (frames, labels) in enumerate(dataloader):
        frames = frames.to(device)
        labels = labels.to(device)
        outputs = model(frames)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), "model.pth")
