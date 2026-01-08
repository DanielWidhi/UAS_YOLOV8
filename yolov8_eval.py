import os
from ultralytics import YOLO
from pathlib import Path


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    
    dataset_yaml = Path(__file__).parent / "data.yaml"
    
    # Path ke model hasil training (best.pt)
    # Secara default tersimpan di runs/detect/yolov8_custom_model/weights/best.pt
    # Path Model - Tambahkan angka 5
    model_path = Path(__file__).parent / "runs" / "detect" / "yolov8_custom_model5" / "weights" / "best.pt"

    # Validasi keberadaan file
    assert dataset_yaml.exists(), f"Dataset YAML tidak ditemukan: {dataset_yaml}"
    assert model_path.exists(), f"Model best.pt tidak ditemukan di {model_path}. Selesaikan training terlebih dahulu!"

    print(f"Memuat model untuk Evaluasi: {model_path}")

    ''' Gunakan model hasil latihan (best.pt) untuk evaluasi '''
    model = YOLO(str(model_path)) 

    # Mulai evaluasi/validasi model
    metrics = model.val(
        data=str(dataset_yaml), # Path ke file YAML dataset
        imgsz=640,              # Ukuran gambar harus sama dengan saat training
        batch=16,               # Ukuran batch untuk evaluasi
        split="val",            # Menggunakan folder 'valid' yang ada di data.yaml
        device="0",             # Gunakan GPU RTX 3050 Anda (device 0)
        verbose=True            # Menampilkan hasil per kelas (Karat & Pengorok)
    )

    # Menampilkan ringkasan hasil di terminal
    print("\n" + "="*40)
    print("HASIL EVALUASI 2 KELAS DAUN KOPI")
    print("="*40)
    print(f"mAP@0.5 (Akurasi Keseluruhan): {metrics.box.map50:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")
    print("="*40)

# WAJIB di Windows untuk menghindari error multiprocessing
if __name__ == "__main__":
    main()