---
title: NER Tanaman Herbal
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Named Entity Recognition pada Teks Tanaman Herbal

Proyek ini merupakan aplikasi berbasis web untuk mengidentifikasi dan mengelompokkan entitas penting dalam abstrak jurnal tanaman herbal berbahasa Indonesia. Sistem menerapkan **Named Entity Recognition (NER)** dengan skema **BIO tagging** untuk mengubah informasi yang terdapat dalam teks menjadi data yang lebih terstruktur.

Aplikasi menerima judul jurnal, nama penulis, tahun terbit, dan abstrak sebagai masukan. Hasil analisis disajikan dalam bentuk penanda berwarna pada teks, ringkasan berdasarkan kategori, serta tabel yang memuat setiap entitas yang terdeteksi.

## Cakupan Entitas

Sistem dirancang untuk mengenali delapan kategori entitas berikut:

| Label        | Kategori                         |
| ------------ | -------------------------------- |
| `HERB`       | Tanaman herbal                   |
| `BODY_PART`  | Bagian atau anggota tubuh        |
| `DISEASE`    | Penyakit atau keluhan            |
| `COMPOUND`   | Senyawa atau kandungan aktif     |
| `EFFECT`     | Efek atau aktivitas biologis     |
| `TREATMENT`  | Terapi atau bentuk pengobatan    |
| `METHOD`     | Metode penelitian atau pengujian |
| `POPULATION` | Populasi atau subjek penelitian  |

## Model yang Digunakan

Penelitian ini menggunakan dua model sebagai bahan perbandingan:

- **Bidirectional Long Short-Term Memory (BiLSTM)** sebagai model utama. Model ini mempelajari konteks suatu token berdasarkan urutan kata sebelum dan sesudahnya.
- **Naive Bayes** sebagai model baseline. Model ini mengklasifikasikan token berdasarkan fitur yang telah dipelajari tanpa mempertimbangkan hubungan sekuensial secara langsung.

### Hasil Evaluasi

| Model       | Precision | Recall | F1-score |
| ----------- | --------: | -----: | -------: |
| BiLSTM      |    0.8333 | 0.8223 |   0.8278 |
| Naive Bayes |    0.4230 | 0.8684 |   0.5689 |

Berdasarkan hasil tersebut, BiLSTM menghasilkan keseimbangan precision dan recall yang lebih baik, sedangkan Naive Bayes memperoleh recall lebih tinggi dengan precision yang lebih rendah.

## Komponen Proyek

```text
Ner-Skripsi/
├── app.py
├── models/
│   ├── bilstm_ner_weights.weights.h5
│   ├── naive_bayes_ner_model.pkl
│   └── ner_mappings.pkl
├── templates/
│   ├── about.html
│   ├── analyze.html
│   └── index.html
├── static/
│   ├── css/
│   │   ├── home.css
│   │   ├── about.css
│   │   └── analyze.css
│   └── images/
├── requirements.txt
├── Dockerfile
└── README.md
```

Komponen utama dalam proyek ini meliputi:

- `app.py` untuk menjalankan aplikasi, memuat model, dan menangani proses prediksi.
- `models/` untuk menyimpan bobot BiLSTM, model Naive Bayes, dan konfigurasi pemetaan NER.
- `templates/` dan `static/` untuk membentuk antarmuka web serta menyediakan aset pendukung.
- `requirements.txt` dan `Dockerfile` untuk mendefinisikan lingkungan aplikasi.

## Teknologi

Aplikasi dikembangkan menggunakan Python, Flask, TensorFlow/Keras, Scikit-learn, NumPy, HTML, dan CSS.

## Ruang Lingkup Penelitian

Data penelitian berfokus pada sepuluh tanaman herbal, yaitu temulawak, jahe, kunyit, mengkudu, lengkuas, sambiloto, daun sirih, pegagan, kelor, dan kumis kucing.

Proyek ini dikembangkan untuk keperluan akademik dan masih bersifat eksperimental. Hasil prediksi dapat digunakan sebagai pendukung analisis awal, tetapi tidak dimaksudkan untuk menggantikan anotasi manual, penilaian ahli, atau pertimbangan medis.


## Demo
http://103.93.129.77/
## Pengembang

**Reza Alfiansyah** — Program Studi Informatika, Universitas Gunadarma
