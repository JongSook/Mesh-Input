import os
import trimesh
import torch
from torch.utils.data import Dataset
import numpy as np

# Load mesh
def load_and_process_mesh(path):
    mesh = trimesh.load_mesh(path, process=True)
    mesh.apply_translation(-mesh.centroid)
    if mesh.scale != 0:
        mesh.apply_scale(1.0 / mesh.scale)
    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int64)

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

# Torch Dataset for mesh input
class MeshInputDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        vertices, faces = load_and_process_mesh(path)
        vertices = torch.tensor(vertices, dtype=torch.float32)
        faces = torch.tensor(faces, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        return vertices, faces, label

# Usage example
if __name__ == "__main__":
    data_path = "your_dataset_directory"  # Replace with your actual path
    file_list = create_file_list_from_directory(data_path)
    dataset = MeshInputDataset(file_list)

    # Load one item
    vertices, faces, label = dataset[0]
    print("Vertices shape:", vertices.shape)
    print("Faces shape:", faces.shape)
    print("Label:", label.item())
