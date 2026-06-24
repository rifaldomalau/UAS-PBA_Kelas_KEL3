## 📅 Rencana Kerja Step-by-Step Projek UAS ABSA

### ✅ 📁 FASE 1: Amankan Data & Eksplorasi (Minggu Ini)

- ✅ **Step 1: Kumpulkan Bahan dari UTS**
  Ambil 3 file utama dari repositori UTS Anda: dataset mentah (`KelpX_dataset_1.csv`), dataset bersih (`KelpX_dataset_2.csv`), dan dataset anotasi (`KelpX_dataset_anotasi.jsonl`). Masukkan semuanya ke folder `project/dataset/`.

- ✅ **Step 2: Potong Data (Data Splitting)**
  Jalankan skrip pemisah data (_Train, Validation, Test_) secara konsisten untuk data Multilabel maupun NER. Pastikan rasionya aman (misal 70:15:15).

- ✅ **Step 3: Kerjakan EDA (Exploratory Data Analysis)**
  Buka Jupyter Notebook, buat visualisasi menarik:

- Grafik panjang review dan jumlah kata.

- Grafik distribusi 13 label multilabel.

- _Correlation matrix_ antar label (untuk melihat apakah ada aspek yang sering muncul bersamaan).

- Grafik jumlah entitas/tag NER (berapa banyak kata berlabel `PRODUCT`, `PRICE`, dll.).

---

### ✅ 🤖 FASE 2: Eksekusi Proyek A – Klasifikasi Multilabel (Target: Nilai 95)

Di fase ini, kita fokus mengajari AI agar bisa menebak kategori besar dari sebuah review.

- ✅ **Step 4: Preprocessing & Penanganan Imbalance**
  Lakukan pembersihan teks (_case folding_, hapus emoji/karakter aneh). Karena ada label yang jarang muncul, siapkan strategi penanganan data tidak seimbang (_label imbalance_).

- ✅ **Step 5: Adu Representasi Teks (Feature Engineering)**
  Ubah teks menjadi angka dengan **dua cara berbeda** untuk dibandingkan, misalnya menggunakan **TF-IDF** klasik melawan **Transformer Embedding (IndoBERT)**.

- ✅ **Step 6: Pelatihan Model Klasifikasi**
  Latih minimal **dua algoritma** (misalnya SVM dan Random Forest) yang dikombinasikan dengan teknik _Problem Transformation_ (seperti _Classifier Chains_ atau _Binary Relevance_). Lakukan _hyperparameter tuning_ agar hasilnya optimal.

- ✅ **Step 7: Evaluasi & Cari Kesalahan**
  Uji semua model ke _test set_. Tampilkan metrik wajib: _Classification Report_, _Micro/Macro F1-score_, _Hamming Loss_, dan _Subset Accuracy_. Jangan lupa buat analisis kesalahan (_error analysis_) dari label yang sering salah tebak. Simpan model terbaiknya!

---

### ✅ 🏷️ FASE 3: Eksekusi Proyek B – Named Entity Recognition (NER)

Di fase ini, kita beralih fokus untuk mengajari AI mendeteksi kata-kata spesifik di dalam kalimat.

- ✅ **Step 8: Tokenisasi & Konversi Tag BIO/BILOU**
  Ubah koordinat _span_ karakter dari UTS menjadi label per kata/token. Jika menggunakan IndoBERT, pastikan _alignment_ aman karena IndoBERT suka memotong kata menjadi _subwords_ (misal: "gepreknya" menjadi "ge", "##prek", "##nya").

- ✅ **Step 9: Setup Arsitektur Deep Learning**
  Bangun model NER berbasis _Deep Learning_. Anda bisa memilih arsitektur seperti **BiLSTM-CRF** atau langsung memakai **IndoBERT Token Classification**. Gunakan teknik _Early Stopping_ saat latihan agar model tidak _overfitting_.

- ✅ **Step 10: Evaluasi Token & Span Level**
  Hitung nilai _Precision_, _Recall_, dan _F1-score_. Analisis kesalahan model, terutama di mana model salah menentukan batas kata (_error boundary_) atau salah memberikan label entitas. Simpan model NER terbaik ini.

---

### ✅ 💻 FASE 4: Deployment ke Aplikasi Web Streamlit

Waktunya menyatukan kedua model hebat kita ke dalam satu wadah visual.

- **Step 11: Coding Aplikasi Streamlit (`KelpX_app.py`)**
  Buat aplikasi web dengan dua menu utama:

- **Tab 1 (Prediksi Multilabel):** Pengguna mengetik review ➡️ Muncul daftar label beserta skor keyakinannya (_confidence score_).

- **Tab 2 (Prediksi NER):** Pengguna mengetik review ➡️ Kata-kata di dalam review langsung otomatis terwarnai (_highlighted_) sesuai jenis aspeknya.

- **Step 12: Jembatan Logika (Korelasi Multilabel & NER)**
  Tulis penjelasan di aplikasi web Anda yang menunjukkan hubungan logis kedua output tersebut. Contoh: Jika model Multilabel mendeteksi `PRICE_NEGATIVE`, aplikasi harus bisa menjelaskan bahwa kesimpulan itu diambil karena model NER berhasil mendeteksi kata "mahal" sebagai entitas `PRICE`.

---

### ⏳ 📝 FASE 5: Pembuatan Laporan & Pengumpulan Berkas (Finishing)

Langkah terakhir untuk memastikan semua administrasi nilai aman.

- **Step 13: Dokumentasi Kode & Laporan**
  Ekspor seluruh Jupyter Notebook Anda menjadi file PDF. Buat laporan resmi (`.pdf` / `.docx`) yang berisi penjelasan dataset, alur eksperimen, tabel perbandingan performa model, analisis kesalahan, screenshot aplikasi web, serta tabel pembagian tugas kelompok.

- **Step 14: Rekam Video Demo App**
  Buat video berdurasi 5–10 menit. Tunjukkan cara kerja aplikasi Streamlit Anda secara langsung menggunakan teks review baru, dan jelaskan hubungan antara hasil multilabel dan NER-nya. Masukkan link video ke file `KelpX_anggota.txt`.

- **Step 15: Bungkus ZIP & Upload**
  Tata semua file sesuai panduan struktur folder, lalu kompres menjadi file ZIP dengan format nama resmi: `Projek UAS PBA_Kelas_KEL.zip`. Terakhir, pastikan **semua** anggota kelompok mengunggah file ZIP yang sama ke situs kuliah agar tidak mendapat nilai 0!
