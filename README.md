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
Precision : 0.5695
Recall    : 0.6719
F1-score  : 0.6165
```

### 2. Naive Bayes

Naive Bayes digunakan sebagai model pembanding atau baseline. Model ini melakukan klasifikasi berdasarkan fitur token, tetapi memiliki keterbatasan dalam memahami hubungan sekuensial antar-token.

Hasil evaluasi model Naive Bayes:

```text
Precision : 0.2596
Recall    : 0.7891
F1-score  : 0.3907
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
Project/
├── app.py
├── models/
│   ├── bilstm_ner_weights.weights.h5
│   ├── naive_bayes_ner_model.pkl
│   └── ner_mappings.pkl
├── templates/
│   └── index.html
├── static/
│   └── img/
│       ├── hero-ner.png
│       ├── jahe.png
│       ├── kunyit.png
│       ├── daun-sirih.png
│       ├── temulawak.png
│       ├── lidah-buaya.png
│       └── sambiloto.png
├── requirements.txt
└── README.md
```

Catatan: folder `static/img/` bersifat opsional. Jika gambar belum tersedia, website tetap dapat berjalan, tetapi tampilan visual tanaman herbal tidak akan muncul.

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

File model tidak disertakan secara otomatis jika ukurannya besar. Pastikan file model sudah diletakkan di folder `models/` sebelum menjalankan aplikasi.

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
