import os
import cv2
from ultralytics import YOLO
from pathlib import Path

# Menghindari Konflik OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # Path Model - Disinkronkan dengan folder hasil training sebelumnya
    # Folder "runs" biasanya ada di direktori tempat Anda menjalankan script
    base_path = Path.cwd() / "runs" / "detect" / "yolov8_custom_model"
    # Path Model - Tambahkan angka 5
    model_path = Path(__file__).parent / "runs" / "detect" / "yolov8_custom_model5" / "weights" / "best.pt"
    
    # Cek apakah model ada
    assert model_path.exists(), f"Model tidak ditemukan: {model_path}. Pastikan sudah selesai training!"

    print(f"Memuat model dari: {model_path}")
    model = YOLO(str(model_path))

    # Buka Kamera laptop (0 = default kamera)
    cap = cv2.VideoCapture(0)

    # Perbaikan: RuntimeError (T kecil) agar program tidak crash saat kamera tidak ada
    if not cap.isOpened():
        raise RuntimeError("Tidak dapat membuka kamera")
    
    print("Program Deteksi 2 Kelas (Karat & Pengorok Daun) Berjalan.")
    print("Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inferensi
        # Modifikasi: device=0 untuk menggunakan RTX 3050 (sangat disarankan daripada CPU)
        results = model.predict(source=frame, conf=0.25, iou=0.5, device=0, verbose=False)

        # Ambil Frame hasil anotasi
        # Model otomatis menampilkan label 'KaratDaunKopi' atau 'PengorokDaunKopi'
        annotated_frame = results[0].plot()
        
        cv2.imshow("YOLOv8 Kopi Disease Real-time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()