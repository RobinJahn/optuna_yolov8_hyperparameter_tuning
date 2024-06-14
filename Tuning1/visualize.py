import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def load_annotations(annotation_file):
    annotations = []
    with open(annotation_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = parts[0]
            if len(parts) == 5:
                # Bounding box format
                bbox = [float(x) for x in parts[1:]]
                annotations.append((class_id, bbox))
    return annotations

def plot_annotations(image_path, annotations, ax, title, color):
    image = Image.open(image_path)
    width, height = image.size
    ax.imshow(image)
    for annotation in annotations:
        class_id, points = annotation
        if len(points) == 4:
            # Bounding box format
            x_center, y_center, box_width, box_height = points
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height
            x_min = x_center - box_width / 2
            y_min = y_center - box_height / 2
            rect = patches.Rectangle((x_min, y_min), box_width, box_height, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
    ax.set_title(title)
    ax.axis('off')

def visualize_annotations(image_dir, label_dir, visual_output_dir, converted_files):
    os.makedirs(visual_output_dir, exist_ok=True)
    
    for filename in converted_files:
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        if os.path.exists(image_path) and os.path.exists(label_path):
            print(f"Visualizing {filename}")
            annotations = load_annotations(label_path)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            plot_annotations(image_path, annotations, ax, "Bounding Box Annotations", 'b')

            visual_path = os.path.join(visual_output_dir, f"bounding_box_{filename}")
            plt.savefig(visual_path)
            plt.close()

# Load the list of converted files from the JSON file
current_dir = os.getcwd()
json_output_path = os.path.join(current_dir, 'converted_files.json')

with open(json_output_path, 'r') as json_file:
    converted_files = json.load(json_file)

# Paths
base_input_dir = os.path.join(current_dir, 'datasets/Assignment2-17') #TODO: replace with project version number
visual_output_dir = os.path.join(current_dir, 'visualizations')

folders = ['train', 'valid', 'test']

for folder in folders:
    image_dir = os.path.join(base_input_dir, folder, 'images')
    label_dir = os.path.join(base_input_dir, folder, 'labels')
    folder_visual_output_dir = os.path.join(visual_output_dir, folder)
    visualize_annotations(image_dir, label_dir, folder_visual_output_dir, converted_files)

print("Visualization of annotations completed.")

