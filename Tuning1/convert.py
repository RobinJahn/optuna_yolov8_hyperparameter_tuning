import os
import json

def convert_polygon_to_bbox(segment):
    x_coords = [point[0] for point in segment]
    y_coords = [point[1] for point in segment]
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    return x_min, y_min, x_max, y_max

def convert_annotations(input_dir, converted_files):
    print(f"Starting conversion for {input_dir} ...")
    
    polygon_count = 0
    file_count = 0
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)
            
            print(f"Processing file: {input_path}")
            file_count += 1

            with open(input_path, 'r') as file:
                lines = file.readlines()
            
            if not lines:
                print(f"No annotations found in {input_path}")
                continue
            
            converted = False
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                class_id = parts[0]
                
                if len(parts) > 5:
                    print(f"Converting polygon to bounding box for line: {line.strip()}")
                    polygon_count += 1
                    segment = [(float(parts[i]), float(parts[i+1])) for i in range(1, len(parts), 2)]
                    x_min, y_min, x_max, y_max = convert_polygon_to_bbox(segment)
                    bbox = [class_id, (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]
                    print(f"Converted bounding box: {bbox}")
                    new_lines.append(' '.join(map(str, bbox)) + '\n')
                    converted = True
                else:
                    new_lines.append(line)
            
            if converted:
                with open(input_path, 'w') as file:
                    file.writelines(new_lines)
                converted_files.append(filename.replace('.txt', '.jpg'))  # Assuming image files have .jpg extension

    print(f"Conversion completed for {input_dir}")
    print(f"Total files processed: {file_count}")
    print(f"Total polygon annotations converted: {polygon_count}")

# Get the current working directory
current_dir = os.getcwd()

# Directories for train, valid, and test sets
base_input_dir = os.path.join(current_dir, 'datasets/Assignment2-16') #TODO: replace with project version number
print("working in", base_input_dir)

converted_files = []
folders = ['train', 'valid', 'test']

for folder in folders:
    input_dir = os.path.join(base_input_dir, folder, 'labels')
    convert_annotations(input_dir, converted_files)

print("All annotation conversions completed.")
print("Files with converted annotations:", converted_files)

# Save the list of converted files to a JSON file
json_output_path = os.path.join(current_dir, 'converted_files.json')
with open(json_output_path, 'w') as json_file:
    json.dump(converted_files, json_file)

