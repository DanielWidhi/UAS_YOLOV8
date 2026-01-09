- # Tahap 1: Persiapan Lingkungan (Environment)

  Pastikan semua "mesin" pendukung sudah terpasang di dalam Virtual Environment (.venv) Anda.

  1. Buka Terminal di VS Code.
  2. Aktifkan .venv:

     " .\venv\Scripts\activate "

  3. Pastikan Library Terinstal (Versi GPU):

     pip install ultralytics
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

- # Tahap 2: Konfigurasi Dataset (data.yaml)

  Ini adalah bagian paling krusial agar tidak ada error "File Not Found". Buka file data.yaml dan pastikan isinya menggunakan Path Absolut:

        # D:/File Kampus/Coding/SMT 5/Computer Vision/UAS_CV_DW_YOLOV8/data.yaml
        train: D:/File Kampus/Coding/SMT 5/Computer Vision/UAS_CV_DW_YOLOV8/train/images
        val: D:/File Kampus/Coding/SMT 5/Computer Vision/UAS_CV_DW_YOLOV8/valid/images
        test: D:/File Kampus/Coding/SMT 5/Computer Vision/UAS_CV_DW_YOLOV8/test/images

        nc: 2
        names: ['KaratDaunKopi', 'PengorokDaunKopi']

- # Tahap 3: Pelatihan Model (yolov8_train.py)

  Jangan jalankan file lain sebelum tahap ini selesai.

  1. Jalankan Script:

     python yolov8_train.py

  2. Pantau Terminal:

     - Pastikan muncul tulisan CUDA:0 (NVIDIA GeForce RTX 3050...). Jika tulisannya CPU, segera hentikan dan instal ulang Torch CUDA.
     - Tunggu sampai muncul Progress Bar (Epoch 1/100, dst).
     - Selesai jika: Muncul pesan Results saved to runs\detect\yolov8_custom_model.

  3. Cek Hasil: Pastikan file best.pt sudah tercipta di folder runs/detect/yolov8_custom_model/weights/.

- # Tahap 4: Evaluasi Akurasi (yolov8_eval.py)

  Setelah punya model best.pt, kita uji seberapa pintar model tersebut.

  1. Pastikan Path di dalam script mengarah ke folder yolov8_custom_model (sesuaikan angkanya jika folder Anda bernama yolov8_custom_model5).

  2. Jalankan Script:

     python yolov8_eval.py

  3. Lihat Hasil: Anda akan melihat tabel Precision, Recall, dan mAP. Jika mAP di atas 0.6 (60%), model sudah layak pakai.

- # Tahap 5: Deteksi Real-Time Kamera (yolov8_deploy.py)

  Tahap akhir untuk pameran atau demonstrasi program.

  1. Pastikan Model Terbaca: Path model harus sesuai dengan hasil training (misal: yolov8_custom_model/weights/best.pt).

  2. Jalankan Script:

     python yolov8_deploy.py

  3. Cara Penggunaan:
     - Arahkan daun kopi ke webcam.
     - Model hanya akan mendeteksi Karat atau Pengorok karena kita sudah mengunci classes=[0, 1].
     - Jika terlalu sensitif (banyak deteksi palsu), naikkan conf=0.5 di kodingan.
     - Tekan 'q' untuk menutup kamera.

  # Tips Tambahan agar Tidak Error Lagi:

        - Jangan memindahkan folder: Jika Anda memindahkan folder UAS_YOLOV8 ke Desktop atau Drive D, Anda WAJIB memperbarui path di data.yaml.

        - Selalu Colok Charger: Laptop gaming seperti LOQ akan mematikan GPU RTX 3050 atau menurunkan performanya jika hanya menggunakan baterai. Hal ini bisa menyebabkan training sangat lambat atau error CUDA.

        - Cek Folder runs: Jika Anda ingin training ulang dari nol, sebaiknya hapus folder runs lama agar YOLO tidak membuat folder baru seperti yolov8_custom_model2, 3, dst yang bisa membingungkan script Eval/Deploy Anda.

  # Urutan Eksekusi Singkat:

        Train (Sampai tamat) → → Eval (Cek akurasi) → → Deploy (Coba Kamera).
