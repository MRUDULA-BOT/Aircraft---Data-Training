import os

label_path = r"D:\archive (1)\PlanesDataset\val\labels"
for filename in os.listdir(label_path):
    if filename.endswith('.txt'):
        with open(os.path.join(label_path, filename), 'r') as file:
            lines = file.readlines()
        
        # Update class IDs to 0
        with open(os.path.join(label_path, filename), 'w') as file:
            for line in lines:
                parts = line.split()
                parts[0] = '0'  # Change class ID to 0
                file.write(" ".join(parts) + '\n')
