---
title: NER Tanaman Herbal
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Named Entity Recognition pada Tanaman Herbal

Project ini merupakan implementasi **Named Entity Recognition (NER)** pada teks jurnal tanaman herbal menggunakan pendekatan **BIO Tagging**. Sistem ini dibuat untuk mengenali entitas penting dalam abstrak jurnal herbal, seperti nama tanaman, penyakit, anggota tubuh, senyawa, efek, terapi, metode, dan populasi.

Project ini menggunakan dua model pembelajaran mesin, yaitu **BiLSTM** sebagai model utama dan **Naive Bayes** sebagai model pembanding. Hasil prediksi model kemudian ditampilkan melalui aplikasi web berbasis **Flask**.

---

## Fitur Utama

- Input data jurnal berupa:
  - Judul jurnal
  - Nama penulis
  - Tahun terbit
  - Abstrak jurnal

- Pilihan model prediksi:
  - BiLSTM
  - Naive Bayes

- Menampilkan hasil prediksi entitas dalam bentuk highlight teks.
- Menampilkan tabel klasifikasi entitas.
- Menampilkan detail entitas yang terdeteksi.
- Menampilkan informasi jurnal yang diproses.
- Tampilan web bertema tanaman herbal.

---

## Entitas yang Dikenali

Sistem ini mengenali beberapa label entitas berikut:

| Label      | Keterangan                       |
| ---------- | -------------------------------- |
| HERB       | Nama tanaman herbal              |
| BODY_PART  | Anggota tubuh                    |
| DISEASE    | Penyakit atau keluhan            |
| COMPOUND   | Senyawa atau kandungan aktif     |
| EFFECT     | Efek atau aktivitas biologis     |
| TREATMENT  | Terapi atau bentuk pengobatan    |
| METHOD     | Metode penelitian atau pengujian |
| POPULATION | Populasi atau subjek penelitian  |

---

## Model yang Digunakan

### 1. BiLSTM

BiLSTM digunakan sebagai model utama karena mampu mempelajari konteks token dari dua arah, yaitu dari token sebelumnya dan token sesudahnya. Model ini lebih sesuai untuk tugas NER berbasis urutan token dan BIO tagging.

Hasil evaluasi model BiLSTM:

```text
Precision : 0.8333
Recall    : 0.8223
F1-score  : 0.8278
```

### 2. Naive Bayes

Naive Bayes digunakan sebagai model pembanding atau baseline. Model ini melakukan klasifikasi berdasarkan fitur token, tetapi memiliki keterbatasan dalam memahami hubungan sekuensial antar-token.

Hasil evaluasi model Naive Bayes:

```text
Precision : 0.4230
Recall    : 0.8684
F1-score  : 0.5689
```

---

## Teknologi yang Digunakan

- Python
- Flask
- TensorFlow / Keras
- NumPy
- Scikit-learn
- HTML
- CSS
- Pickle

---

## Struktur Folder Project

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
├── Dataset_NER_BIO_Tagging_Final.csv
├── NER_BiLSTM_NB_FIXED.ipynb
├── .dockerignore
├── .gitignore
└── README.md
```

Folder `models/`, `templates/`, dan `static/` harus berada sejajar dengan `app.py` agar aplikasi dapat menemukannya saat dijalankan secara lokal maupun melalui Docker.

---

## Instalasi

### 1. Clone repository

```bash
git clone https://github.com/username/nama-repository.git
cd nama-repository
```

### 2. Buat virtual environment

```bash
python -m venv venv
```

Aktifkan virtual environment.

Untuk Windows PowerShell:

```powershell
.\venv\Scripts\activate
```

Untuk Linux/Mac:

```bash
source venv/bin/activate
```

### 3. Install dependency

```bash
pip install -r requirements.txt
```

Jika belum memiliki file `requirements.txt`, install library berikut:

```bash
pip install flask tensorflow numpy scikit-learn markupsafe
```

---

## File Model

Aplikasi membutuhkan tiga file model berikut di dalam folder `models/`:

```text
models/
├── bilstm_ner_weights.weights.h5
├── naive_bayes_ner_model.pkl
└── ner_mappings.pkl
```

Keterangan:

| File                            | Fungsi                                                 |
| ------------------------------- | ------------------------------------------------------ |
| `bilstm_ner_weights.weights.h5` | Bobot model BiLSTM                                     |
| `naive_bayes_ner_model.pkl`     | Model Naive Bayes                                      |
| `ner_mappings.pkl`              | Mapping token, label, threshold, dan konfigurasi model |

Ketiga file model disertakan dalam repository dan harus tetap berada di folder `models/` agar aplikasi dapat dijalankan dan di-deploy.

---

## Menjalankan Aplikasi

Jalankan perintah berikut dari folder project:

```bash
python app.py
```

Setelah server berjalan, buka browser dan akses:

```text
http://127.0.0.1:5000/
```

---

## Deployment ke Hugging Face Spaces

Repository ini sudah dikonfigurasi sebagai Docker Space dengan port `7860`.

1. Buat Space baru di Hugging Face.
2. Pilih **Docker** sebagai SDK dan **CPU Basic** sebagai hardware.
3. Tambahkan repository Space sebagai remote Git:

```bash
git remote add hf https://huggingface.co/spaces/USERNAME/ner-skripsi
```

4. Untuk deployment pertama, kirim branch lokal `master` ke branch `main` milik Space:

```bash
git push --force hf master:main
```

Ganti `USERNAME` dengan username Hugging Face. Gunakan Hugging Face Access Token dengan izin tulis ketika Git meminta password. Deployment berikutnya cukup menggunakan `git push hf master:main`.

---

## Cara Penggunaan

1. Masukkan judul jurnal.
2. Masukkan nama penulis.
3. Masukkan tahun terbit.
4. Masukkan abstrak jurnal tanaman herbal.
5. Pilih model:
   - BiLSTM
   - Naive Bayes

6. Klik tombol **Proses NER**.
7. Sistem akan menampilkan:
   - Output teks dengan highlight entitas
   - Tabel klasifikasi entitas
   - Detail entitas terdeteksi
   - Informasi jurnal

---

## Contoh Entitas

Contoh input:

```text
Rebusan jahe digunakan untuk mengurangi keluhan mual muntah pada ibu hamil.
```

Contoh entitas yang dapat dikenali:

| Entitas     | Label      |
| ----------- | ---------- |
| jahe        | HERB       |
| mual muntah | DISEASE    |
| ibu hamil   | POPULATION |

---

## Catatan Pengembangan

Model BiLSTM pada project ini menggunakan mekanisme load weights, bukan load model penuh `.keras` atau `.h5`. Hal ini dilakukan untuk menghindari masalah kompatibilitas versi TensorFlow/Keras, terutama error pada konfigurasi layer seperti `quantization_config`.

Aplikasi membangun ulang arsitektur BiLSTM di `app.py`, kemudian memuat bobot dari file:

```text
bilstm_ner_weights.weights.h5
```

---

## Keterbatasan Sistem

Sistem ini masih memiliki beberapa keterbatasan:

- Dataset masih terbatas.
- Distribusi label tidak seimbang.
- Beberapa label memiliki jumlah data yang kecil.
- Model masih dapat menghasilkan kesalahan prediksi.
- Beberapa entitas dapat tidak terdeteksi jika konteks kalimat terlalu pendek.
- Naive Bayes memiliki keterbatasan dalam memahami hubungan urutan token.

Oleh karena itu, hasil prediksi pada website sebaiknya dipahami sebagai hasil prediksi model, bukan sebagai hasil anotasi yang sepenuhnya pasti benar.

---

## Disclaimer

Project ini dikembangkan menggunakan dataset NER tanaman herbal yang berfokus pada 10 jenis tanaman, yaitu temulawak, jahe, kunyit, mengkudu, lengkuas, sambiloto, daun sirih, pegagan, kelor, dan kumis kucing.f1

Dataset disusun menggunakan pendekatan BIO Tagging berdasarkan teks jurnal atau artikel ilmiah berbahasa Indonesia. Oleh karena itu, performa model bergantung pada kualitas dataset, konsistensi pelabelan, jumlah data, serta distribusi label pada setiap entitas.

Apabila aplikasi masih mengalami error, hasil prediksi belum sesuai, atau terdapat kendala saat memuat file model, pengguna dapat menjalankan ulang proses training menggunakan dataset terbaru yang telah dikoreksi. Training ulang diperlukan apabila terdapat perubahan pada token, label, mapping, atau format dataset.

Project ini masih bersifat akademik dan eksperimental, sehingga hasil prediksi model sebaiknya digunakan sebagai bantuan analisis awal, bukan sebagai hasil anotasi final yang sepenuhnya pasti benar.

---

## Tujuan Penelitian

Project ini dibuat sebagai bagian dari penelitian mengenai penerapan Named Entity Recognition pada teks tanaman herbal. Sistem ini bertujuan untuk membantu proses ekstraksi informasi dari teks jurnal herbal agar entitas penting dapat ditampilkan secara lebih terstruktur.

---

## Author

**Reza Alfiansyah**
Program Studi Informatika
Universitas Gunadarma

---

## License

Project ini dibuat untuk kebutuhan akademik dan penelitian.
