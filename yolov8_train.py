import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
from pathlib import Path

def main():

    # Path ini akan otomatis membaca 2 kelas (Karat & Pengorok) sesuai isi data.yaml Anda
    dataset_yaml = Path(__file__).parent / "data.yaml"
    assert dataset_yaml.exists(), f"Dataset YAML tidak ditemukan: {dataset_yaml}"


    model = YOLO("yolov8s.pt")  # Gunakan model YOLOv8s sebagai dasar

    # Mulai pelatihan model
    model.train(
        data=str(dataset_yaml), # Path ke file YAML dataset
        epochs=100,               # Jumlah epoch pelatihan
        imgsz=640,                # Ukuran gambar input
        batch=16,                 # Ukuran batch (RTX 3050 sangat aman dengan batch 16)
        patience=10,              # Early stopping jika tidak ada peningkatan selama 10 epoch
        lr0=0.001,                # Learning rate awal
        lrf=0.01,                 # Learning rate akhir
        optimizer="AdamW",        # Optimizer yang digunakan
        weight_decay=0.001,       # Weight decay untuk regularisasi
        device=0,               # Gunakan GPU RTX 3050 (device 0)
        name="yolov8_custom_model" # Nama untuk menyimpan hasil pelatihan
    )

# --- BAGIAN WAJIB UNTUK WINDOWS ---
# Baris ini ditambahkan agar tidak terjadi error 'multiprocessing' pada laptop LOQ Anda
if __name__ == "__main__":
    main()