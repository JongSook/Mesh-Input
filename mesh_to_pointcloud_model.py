import os
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d

# Mesh to pointcloud
def mesh_to_point_cloud(path, num_points=1024):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points).astype(np.float32)

# Load dataset from folder structure
def create_file_list_from_directory(data_path):
    file_list = []
    label_map = {name: idx for idx, name in enumerate(sorted(os.listdir(data_path)))}
    for class_name, label in label_map.items():
        class_dir = os.path.join(data_path, class_name)
        for file in os.listdir(class_dir):
            if file.endswith('.obj') or file.endswith('.stl') or file.endswith('.ply'):
                file_path = os.path.join(class_dir, file)
                file_list.append((file_path, label))
    return file_list

# Torch Dataset for point cloud input
class PointCloudInputDataset(Dataset):
    def __init__(self, file_list, num_points=1024):
        self.file_list = file_list
        self.num_points = num_points

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        point_cloud = mesh_to_point_cloud(path, self.num_points)
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return point_cloud, label

# Pointcloud classification
class SimplePointCloudClassifier(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, num_classes=2):
        super(SimplePointCloudClassifier, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.fc1(x)         # (B, N, H)
        x = F.relu(x)
        x = self.fc2(x)         # (B, N, H)
        x = F.relu(x)
        x = x.mean(dim=1)       # Global average pooling (B, H)
        out = self.classifier(x)
        return out

# Usage example
if __name__ == "__main__":
    data_path = "your_dataset_directory"  # Replace with your actual path
    file_list = create_file_list_from_directory(data_path)
    dataset = PointCloudInputDataset(file_list)

    # Load one item and pass through model
    point_cloud, label = dataset[0]  # shape (N, 3)
    point_cloud = point_cloud.unsqueeze(0)  # Add batch dimension (1, N, 3)

    model = SimplePointCloudClassifier(in_channels=3, num_classes=len(set(lbl for _, lbl in file_list)))
    output = model(point_cloud)
    print("Model output logits:", output)
    print("Predicted class:", torch.argmax(output).item())
