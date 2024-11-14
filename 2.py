import os
from PIL import Image

def resize_and_pad_images(image_folder, target_size=(640, 640)):
    print("Starting to process images...")  # This will print to the shell
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path)
            
            # Resize image while maintaining aspect ratio
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Create a new image with target size and paste the resized image into it
            new_img = Image.new("RGB", target_size, (0, 0, 0))  # black background
            new_img.paste(img, ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2))
            
            # Save the image
            new_img.save(img_path)
            
            print(f"Processed: {filename}")  # Prints which image is being processed

# Call the function with the path to your image folder
resize_and_pad_images('D:\\archive (1)\\PlanesDataset\\images')
