import cv2
import numpy as np
import os

# Fungsi untuk membuat gambar sintetis
def create_synthetic_image(density, image_size=(256, 256)):
    image = np.ones(image_size, dtype=np.uint8) * 255  # Create a white image

    num_buildings = density * 20  # More density, more buildings
    for _ in range(num_buildings):
        x, y = np.random.randint(0, image_size[0]), np.random.randint(0, image_size[1])
        w, h = np.random.randint(10, 30), np.random.randint(10, 30)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)  # Draw black buildings

    return image

# Buat folder untuk menyimpan dataset
dataset_path = 'synthetic_dataset'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Buat dataset dengan berbagai tingkat kepadatan penduduk
densities = [1, 2, 3]  # 1: rendah, 2: menengah, 3: tinggi
labels = []

for i, density in enumerate(densities):
    for j in range(10):  # Buat 10 gambar untuk setiap tingkat kepadatan
        image = create_synthetic_image(density)
        image_path = os.path.join(dataset_path, f'image_{i}_{j}.png')
        cv2.imwrite(image_path, image)
        labels.append((image_path, density - 1))  # Label 0 untuk rendah, 1 untuk menengah, 2 untuk tinggi

# Simpan label ke file teks
with open(os.path.join(dataset_path, 'labels.txt'), 'w') as f:
    for label in labels:
        f.write(f'{label[0]},{label[1]}\n')

print('Dataset dan label berhasil dibuat.')
