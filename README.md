---
title: NER Tanaman Herbal
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# NER Tanaman Herbal

Aplikasi web untuk mengenali entitas penting dalam abstrak jurnal tanaman herbal berbahasa Indonesia. Teks diproses menggunakan skema **BIO tagging**, kemudian setiap entitas yang ditemukan ditandai pada teks dan dirangkum dalam tabel agar hasilnya lebih mudah dibaca.

Proyek ini membandingkan dua pendekatan: **Bidirectional Long Short-Term Memory (BiLSTM)** sebagai model utama dan **Naive Bayes** sebagai model baseline. Antarmuka aplikasinya dibangun dengan Flask.

## Entitas yang dikenali

| Label        | Keterangan                       |
| ------------ | -------------------------------- |
| `HERB`       | Nama tanaman herbal              |
| `BODY_PART`  | Bagian atau anggota tubuh        |
| `DISEASE`    | Penyakit atau keluhan            |
| `COMPOUND`   | Senyawa atau kandungan aktif     |
| `EFFECT`     | Efek atau aktivitas biologis     |
| `TREATMENT`  | Terapi atau bentuk pengobatan    |
| `METHOD`     | Metode penelitian atau pengujian |
| `POPULATION` | Populasi atau subjek penelitian  |

## Fitur aplikasi

- Memproses judul, penulis, tahun terbit, dan abstrak jurnal.
- Menyediakan pilihan model BiLSTM dan Naive Bayes.
- Menandai entitas secara langsung pada teks hasil prediksi.
- Merangkum entitas berdasarkan kategori dan menampilkan detail setiap temuan.
- Menyertakan informasi jurnal serta waktu pemrosesan pada hasil analisis.

## Model dan hasil evaluasi

BiLSTM membaca urutan token dari dua arah sehingga dapat mempertimbangkan konteks sebelum dan sesudah sebuah token. Naive Bayes digunakan sebagai pembanding yang lebih sederhana karena melakukan klasifikasi berdasarkan fitur token tanpa memodelkan hubungan sekuensial secara langsung.

| Model       | Peran       | Precision | Recall | F1-score |
| ----------- | ----------- | --------: | -----: | -------: |
| BiLSTM      | Model utama |    0.8333 | 0.8223 |   0.8278 |
| Naive Bayes | Baseline    |    0.4230 | 0.8684 |   0.5689 |

Angka di atas merupakan hasil evaluasi pada data penelitian yang digunakan saat model dikembangkan. Nilainya dapat berubah apabila dataset, pembagian data, atau konfigurasi pelatihan diperbarui.

## Teknologi

- Python 3.10
- Flask dan Gunicorn
- TensorFlow/Keras
- Scikit-learn
- NumPy
- HTML dan CSS

## Menjalankan aplikasi secara lokal

### 1. Kloning repositori

```bash
git clone https://github.com/Rezaalfiah/Ner-Skripsi.git
cd Ner-Skripsi
```

### 2. Siapkan virtual environment

```bash
python -m venv venv
```

Aktifkan environment sesuai sistem operasi yang digunakan.

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Linux atau macOS:

```bash
source venv/bin/activate
```

### 3. Pasang dependensi

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Jalankan server

```bash
python app.py
```

Setelah server aktif, buka `http://127.0.0.1:5000` melalui browser.

## Menjalankan dengan Docker

Bangun image dan jalankan container dari direktori utama proyek:

```bash
docker build -t ner-tanaman-herbal .
docker run --rm -p 7860:7860 ner-tanaman-herbal
```

Aplikasi dapat diakses melalui `http://127.0.0.1:7860`.

## Cara menggunakan

1. Isi judul jurnal, nama penulis, tahun terbit, dan abstrak yang akan dianalisis.
2. Pilih model BiLSTM atau Naive Bayes.
3. Tekan **Proses NER**.
4. Periksa teks yang telah diberi penanda serta tabel entitas di bagian hasil.

Contoh teks masukan:

```text
Rebusan jahe digunakan untuk mengurangi keluhan mual muntah pada ibu hamil.
```

Contoh entitas yang dapat dikenali:

| Teks        | Label        |
| ----------- | ------------ |
| jahe        | `HERB`       |
| mual muntah | `DISEASE`    |
| ibu hamil   | `POPULATION` |

Hasil aktual tetap bergantung pada konteks kalimat dan kemampuan model dalam mengenali pola yang pernah dipelajari.

## Struktur repositori

```text
Ner-Skripsi/
├── app.py
├── Dockerfile
├── requirements.txt
├── models/
│   ├── bilstm_ner_weights.weights.h5
│   ├── naive_bayes_ner_model.pkl
│   └── ner_mappings.pkl
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── images/
├── .dockerignore
├── .gitignore
└── README.md
```

Direktori `models`, `templates`, dan `static` harus tetap berada sejajar dengan `app.py` karena lokasi tersebut digunakan langsung oleh aplikasi.

### Berkas model

| Berkas                          | Kegunaan                                                     |
| ------------------------------- | ------------------------------------------------------------ |
| `bilstm_ner_weights.weights.h5` | Bobot model BiLSTM                                           |
| `naive_bayes_ner_model.pkl`     | Model Naive Bayes yang telah dilatih                         |
| `ner_mappings.pkl`              | Kosakata, label, threshold, dan konfigurasi pemrosesan model |

Arsitektur BiLSTM dibangun kembali di `app.py`, kemudian bobotnya dimuat dari `bilstm_ner_weights.weights.h5`. Pendekatan ini digunakan agar proses pemuatan model lebih konsisten dengan versi TensorFlow/Keras pada aplikasi.

## Deployment ke Hugging Face Spaces

Repositori ini telah dikonfigurasi sebagai **Docker Space** dan menggunakan port `7860`.

1. Buat Space baru di Hugging Face dan pilih **Docker** sebagai SDK.
2. Tambahkan Space sebagai remote Git:

   ```bash
   git remote add hf https://huggingface.co/spaces/USERNAME/ner-skripsi
   ```

3. Kirim branch lokal ke branch `main` pada Space:

   ```bash
   git push hf master:main
   ```

Ganti `USERNAME` dengan nama pengguna Hugging Face. Saat diminta kredensial, gunakan Hugging Face Access Token yang memiliki izin tulis.

## Ruang lingkup dan keterbatasan

Dataset penelitian berfokus pada sepuluh tanaman: temulawak, jahe, kunyit, mengkudu, lengkuas, sambiloto, daun sirih, pegagan, kelor, dan kumis kucing. Karena cakupan data serta distribusi label masih terbatas, model dapat melewatkan entitas atau memberikan label yang kurang tepat, terutama pada istilah dan susunan kalimat yang belum banyak muncul dalam data pelatihan.

Aplikasi ini ditujukan untuk penelitian dan eksplorasi awal. Hasil prediksi tidak menggantikan anotasi manual, penilaian ahli, ataupun pertimbangan medis. Pelatihan ulang diperlukan apabila kosakata, label, pemetaan, format dataset, atau data penelitian mengalami perubahan.

## Tentang penelitian

Proyek ini dikembangkan sebagai bagian dari penelitian mengenai penerapan Named Entity Recognition pada teks tanaman herbal. Tujuannya adalah membantu mengekstrak informasi dari abstrak jurnal berbahasa Indonesia dan menyajikannya dalam bentuk yang lebih terstruktur.

**Reza Alfiansyah** — Program Studi Informatika, Universitas Gunadarma

## Lisensi

Proyek ini dibuat untuk keperluan akademik dan penelitian. Belum ada lisensi sumber terbuka khusus yang disertakan dalam repositori.
