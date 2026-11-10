## 1. Nama Aplikasi

**CustomerMap** (subjudul: *Segmentasi & Prediksi Pelanggan*)

---

## 2. Deskripsi Aplikasi

CustomerMap adalah aplikasi web untuk mengelompokkan pelanggan layanan pesan-antar makanan ke dalam tiga segmen perilaku, yaitu **Pelanggan Loyal**, **Pelanggan Reguler**, dan **Pelanggan Pasif**. Segmentasi dilakukan dengan algoritma *K-Means clustering* pada tiga fitur RFM yang dihitung dari data order dalam jendela 30 hari. Hasil segmentasi membantu pengelola layanan menentukan strategi promosi dan retensi yang berbeda untuk setiap kelompok pelanggan.

Aplikasi dibangun dengan **Python** dan **Streamlit**. Model dikembangkan dan diverifikasi di **RapidMiner (Altair AI Studio)**, lalu direplikasi di Python agar dapat dipakai sebagai aplikasi. Hasil kedua alat dicocokkan pada ukuran yang dapat dibandingkan (Davies-Bouldin Index dan WCSS).

**Data.** Dataset berisi order pelanggan dengan kolom `customer_id`, `order_date`, dan `order_total`. Fitur dihitung per pelanggan dari order pada jendela 1–30 Oktober 2025:

| Fitur | Arti | Perlakuan |
|---|---|---|
| Recency | Jumlah hari sejak order terakhir (1 = paling baru, 30 = paling lama di jendela) | Dipakai apa adanya |
| Frequency | Jumlah order dalam jendela | `ln(x + 1)` |
| Monetary | Total belanja dalam jendela | `ln(x + 1)` |

**Ringkasan hasil model** (k = 3):

| Ukuran | Nilai |
|---|---|
| Customer dianalisis (setelah 100 outlier dibuang) | 4.879 |
| Silhouette Score | 0,4466 |
| Davies-Bouldin Index | 0,7434 |
| WCSS / Inertia | 191,05 |
| Avg. within centroid distance (jarak biasa) | 0,1808 |

**Komposisi dan profil segmen** (centroid dalam satuan asli):

| Segmen | Jumlah (%) | Total belanja (satuan dataset) | Order | Recency (hari) |
|---|---|---|---|---|
| Pelanggan Loyal | 835 (17,1%) | 534,57 | 14,0 | 2,5 |
| Pelanggan Reguler | 2.534 (51,9%) | 64,74 | 2,0 | 6,5 |
| Pelanggan Pasif | 1.510 (30,9%) | 37,76 | 1,3 | 21,1 |


---

## 3. Penjelasan Fitur

### 3.1 Simulator Prediksi
Memprediksi segmen **satu pelanggan** dari ringkasan aktivitasnya dalam 30 hari terakhir.

| Isian | Aturan |
|---|---|
| Frekuensi | Bilangan bulat, minimal 1 |
| Monetary (Rp) | Angka, minimal 0 |
| Recency | Bilangan bulat 1–30 (hari sejak order terakhir) |

Keluaran yang ditampilkan:
- **Segmen hasil prediksi** beserta persentase kemiripan terhadap segmen tersebut.
- **Faktor penentu**: fitur yang mendukung atau condong ke segmen lain, dibanding segmen terdekat kedua.
- **Kartu RFM** (Frekuensi, Monetary, Recency) yang dibandingkan dengan rata-rata segmen.
- **Donat kedekatan** ke ketiga segmen dan **grafik posisi pelanggan** dibanding centroid tiap segmen.
- **Rekomendasi tindakan** sesuai segmen dan **tabel profil rata-rata** tiap segmen.
- **Validasi**: peringatan bila ada isian kosong, dan peringatan bila nilai berada di luar rentang data training (hasil prediksi kurang dapat diandalkan).

### 3.2 Evaluasi Model
Menguji model terhadap dataset yang diunggah (CSV dengan kolom `customer_id`, `order_date`, `order_total`).

- **Tab Hasil Dataset**: jumlah baris, customer, outlier, dan customer yang dianalisis; tabel hasil clustering per customer yang dapat **diunduh sebagai CSV**; serta *scatter plot* dengan pilihan sumbu X dan Y.
- **Tab Metrik Evaluasi**: Silhouette Score, Davies-Bouldin Index, Avg. within centroid distance, Inertia (SSE), distribusi customer per segmen, silhouette per segmen, dan pengaruh tiap fitur terhadap pembagian cluster.
- **Unggah Ulang**: mengganti dataset tanpa membuka ulang aplikasi.
- Pesan kesalahan yang jelas bila kolom wajib hilang atau dataset tidak memuat order pada jendela 30 hari.

### 3.4 Pelatihan Model
Skrip pelatihan menghasilkan satu berkas `kmeans_model_v2.pkl` berisi model K-Means, *scaler*, daftar fitur, label segmen, konfigurasi, dan metrik. Alur pemrosesannya:

| Tahap | Keterangan |
|---|---|
| 1. Filter jendela | Hanya order 1–30 Oktober 2025 |
| 2. Agregasi | Per customer: total belanja, jumlah order, tanggal order terakhir |
| 3. Fitur | Recency (1–30), `ln(x+1)` untuk Frequency dan Monetary |
| 4. Normalisasi sementara | Min-max 0–1 agar jarak outlier adil bagi tiga fitur |
| 5. Deteksi outlier | Jarak ke 20 tetangga terdekat; 100 outlier dibuang |
| 6. Normalisasi final | Min-max 0–1 pada data bersih |
| 7. K-Means | k = 3, 20 percobaan awal, maksimal 100 iterasi, seed 2001 |
| 8. Pelabelan | Loyal, Reguler, dan Pasif ditentukan otomatis dari posisi centroid |
| 9. Evaluasi | Avg. within centroid distance, Silhouette, Davies-Bouldin, Inertia, pengaruh fitur |

Alur yang sama dipakai persis oleh Simulator Prediksi dan Evaluasi Model, sehingga hasil ketiganya konsisten.

### 3.5 Segmen Pelanggan

| Segmen | Ciri | Arah strategi |
|---|---|---|
| Pelanggan Loyal | Sering memesan, belanja tinggi, order terakhir sangat baru | Program loyalitas, poin reward, promo eksklusif, ajakan referral |
| Pelanggan Reguler | Frekuensi dan nilai belanja sedang | Voucher bundling, diskon pembelian berikutnya, target order bulanan |
| Pelanggan Pasif | Jarang memesan, order terakhir sudah lama | Voucher comeback, notifikasi personal, survei alasan berhenti |

---

## 4. Product Requirements Document (PRD)

### 4.1 Ringkasan Produk
CustomerMap adalah alat bantu analitik berbasis web yang mengubah data order mentah menjadi segmen pelanggan yang dapat ditindaklanjuti, dilengkapi simulator untuk memprediksi segmen satu pelanggan dan halaman evaluasi untuk menilai kualitas model.

### 4.2 Latar Belakang dan Masalah
- Layanan pesan-antar menyimpan banyak data order, tetapi perilaku pelanggan belum dikelompokkan secara terstruktur, sehingga promosi cenderung seragam untuk semua pelanggan.
- Pelanggan bernilai tinggi dan pelanggan yang hampir berhenti sulit dibedakan hanya dengan melihat daftar order.
- Versi awal model memakai *tanggal dalam bulan* sebagai ukuran recency sehingga cluster didominasi tanggal order terakhir. Versi sekarang memakai jarak hari dalam jendela 30 hari agar ketiga fitur sama-sama berpengaruh.

### 4.3 Tujuan dan Indikator Keberhasilan

**Tujuan:**
1. Mengelompokkan pelanggan ke dalam segmen yang bermakna dan mudah dijelaskan.
2. Menyediakan simulator prediksi segmen yang mudah dipakai.
3. Menyediakan evaluasi model yang transparan.
4. Menjaga hasil Python konsisten dengan model di RapidMiner.

**Indikator keberhasilan** (ambang ditetapkan tim, dapat disesuaikan):

| Indikator | Target | Hasil saat ini |
|---|---|---|
| Davies-Bouldin Index | < 1,0 | 0,7434 |
| Silhouette Score | ≥ 0,25 (struktur cluster tidak "tidak jelas") | 0,4466 |
| Pengaruh fitur tidak didominasi satu fitur | Tiap fitur < 70% | Maksimum 45,3% (Recency) |
| Kesesuaian dengan RapidMiner | DBI dan WCSS cocok | DBI 0,743 vs 0,7434; WCSS 191,051 vs 191,0512 |
| Pengujian black-box | Seluruh skenario "Sesuai" | *Diisi setelah pengujian* |

### 4.4 Target Pengguna

| Pengguna | Kebutuhan |
|---|---|
| Pengelola layanan / tim pemasaran | Mengetahui segmen pelanggan dan strategi yang cocok |
| Analis data | Menilai kualitas model dan memeriksa hasil per customer |
| Pengembang / peneliti | Melatih ulang model dan membandingkan hasilnya |

### 4.5 Ruang Lingkup

**Dalam lingkup:**
- Segmentasi RFM dengan K-Means (k = 3) pada jendela 30 hari.
- Halaman beranda, Simulator Prediksi, dan Evaluasi Model.
- Skrip pelatihan model dan penyimpanan hasil ke berkas `.pkl`.
- Pengunduhan hasil clustering dalam format CSV.

**Di luar lingkup (saat ini):**
- Autentikasi pengguna dan pengelolaan hak akses.
- Prediksi massal lewat unggah berkas pada Simulator.
- Koneksi langsung ke basis data order.
- Pembedaan status order (*completed* dan *canceled*) pada perhitungan RFM.
- Pelatihan ulang otomatis dari dalam aplikasi.

### 4.6 User Stories

| ID | Sebagai | Saya ingin | Agar |
|---|---|---|---|
| US-01 | Pengelola layanan | memasukkan data satu pelanggan | tahu segmennya dan tindakan yang disarankan |
| US-02 | Pengelola layanan | melihat alasan di balik hasil prediksi | memahami fitur yang paling menentukan |
| US-03 | Analis data | mengunggah dataset order | melihat segmen seluruh pelanggan dan mengunduh hasilnya |
| US-04 | Analis data | melihat Silhouette dan Davies-Bouldin | menilai kualitas cluster |
| US-05 | Analis data | melihat sebaran cluster pada grafik | memeriksa pemisahan antar segmen |
| US-06 | Pengembang | melatih ulang model dengan data baru | memperbarui segmen tanpa mengubah aplikasi |

### 4.7 Kebutuhan Fungsional

| ID | Kebutuhan | Prioritas |
|---|---|---|
| FR-01 | Beranda tampil saat aplikasi pertama dibuka dan memuat tombol **Coba Sekarang** menuju Simulator Prediksi | Wajib |
| FR-02 | Sidebar berisi menu Simulator Prediksi dan Evaluasi Model, dengan menu aktif diberi tanda | Wajib |
| FR-03 | Simulator menerima Frekuensi (bilangan bulat ≥ 1), Monetary (≥ 0), dan Recency (1–30) | Wajib |
| FR-04 | Sistem menampilkan peringatan bila ada isian kosong dan tidak memproses prediksi | Wajib |
| FR-05 | Prediksi memakai model dan scaler yang sama dengan saat pelatihan | Wajib |
| FR-06 | Sistem menampilkan segmen, persentase kemiripan, faktor penentu, dan rekomendasi tindakan | Wajib |
| FR-07 | Sistem memberi peringatan bila input di luar rentang data training | Penting |
| FR-08 | Evaluasi menerima berkas CSV dan memvalidasi kolom `customer_id`, `order_date`, `order_total` | Wajib |
| FR-09 | Evaluasi menampilkan hasil per customer dan menyediakan unduhan CSV | Wajib |
| FR-10 | Evaluasi menampilkan Silhouette, Davies-Bouldin, Avg. within centroid distance, dan Inertia | Wajib |
| FR-11 | Evaluasi menampilkan scatter plot, distribusi segmen, dan pengaruh fitur | Penting |
| FR-12 | Sistem menampilkan pesan kesalahan yang jelas (model tidak ditemukan, kolom hilang, tidak ada order di jendela) | Wajib |
| FR-13 | Skrip pelatihan menyimpan model, scaler, fitur, label segmen, konfigurasi, dan metrik dalam satu berkas `.pkl` | Wajib |

### 4.8 Kebutuhan Non-Fungsional

| Aspek | Kebutuhan |
|---|---|
| Konsistensi | Praproses pada pelatihan, prediksi, dan evaluasi identik |
| Kegunaan | Antarmuka berbahasa Indonesia dengan tata letak kartu dan palet warna yang seragam |
| Keandalan | Aplikasi tidak berhenti saat input salah atau berkas bermasalah, melainkan menampilkan pesan; model dimuat ulang otomatis saat berkas diganti |
| Keamanan | Data diproses lokal dan tidak dikirim ke layanan luar. Berkas `.pkl` hanya boleh dimuat dari sumber tepercaya, karena format pickle dapat menjalankan kode |
| Keterawatan | Konfigurasi (label, warna, parameter) berada di bagian atas tiap berkas; menambah menu cukup menambah satu baris daftar |
| Portabilitas | Berjalan lokal dengan Python dan Streamlit tanpa layanan tambahan |

### 4.9 Alur Pengguna
1. Pengguna membuka aplikasi dan melihat **Beranda**.
2. Pengguna menekan **Coba Sekarang** dan masuk ke **Simulator Prediksi**.
3. Pengguna mengisi Frekuensi, Monetary, dan Recency, lalu menekan **Prediksi Segmen** dan membaca segmen, alasan, serta rekomendasinya.
4. Pengguna membuka **Evaluasi Model** lewat sidebar, mengunggah dataset, lalu memeriksa hasil per customer dan metrik kualitas model.

### 4.10 Asumsi dan Batasan
- Data yang dipakai hanya satu jendela (1–30 Oktober 2025) dengan tanggal acuan tetap. Order setelah tanggal acuan menghasilkan recency negatif dan peringatan di luar rentang.
- Seluruh order dihitung tanpa membedakan status. Order yang dibatalkan ikut menambah Frequency dan Monetary, sehingga ukuran ini lebih menggambarkan keaktifan memesan daripada pendapatan.
- Satuan `order_total` pada data pelatihan harus sama dengan data yang dipakai di aplikasi (pengaturan `ORDER_TOTAL_UNIT` pada Simulator).
- Nilai Silhouette 0,4466 termasuk kategori lemah (cluster saling tumpang tindih), sehingga kecil kemungkinan ada batas alami antar segmen.
- Kurva elbow tidak memiliki siku yang tajam. Pemilihan k = 3 didasarkan pada penurunan WCSS terbesar dari k = 2 ke k = 3 dan kemudahan menafsirkan tiga segmen.
- Label segmen ditentukan dari posisi centroid, bukan dari data berlabel, dan perlu divalidasi oleh pihak layanan.

### 4.11 Risiko dan Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Data baru lebih baru dari tanggal acuan | Recency negatif, hasil kurang andal | Perbarui tanggal acuan dan latih ulang berkala |
| Satuan `order_total` salah | Monetary jauh dari rentang training | Periksa teks rentang pada form dan peringatan di luar rentang |
| Pelanggan di perbatasan antar segmen | Hasil prediksi berubah dengan perubahan kecil | Tampilkan persentase kemiripan, bukan hanya label |
| Berkas `.pkl` tidak tepercaya | Eksekusi kode berbahaya | Muat hanya berkas hasil pelatihan sendiri |
| Silhouette lemah | Segmen kurang terpisah | Bandingkan algoritma dan nilai k lain, validasi dengan pakar bisnis |

### 4.12 Rencana Pengembangan Lanjutan
- Memperluas periode data atau memakai panel pelanggan per bulan.
- Membandingkan model dengan hanya order *completed* dan menambah fitur seperti rata-rata nilai per order atau tingkat pembatalan.
- Menyediakan prediksi massal dan koneksi basis data.
- Jendela RFM bergulir dengan pelatihan ulang terjadwal serta pencatatan versi model.
- Pengujian kegunaan dengan pengguna nyata dan pengujian otomatis.

---

## 5. Pembagian Peran Tim

| No | Nama | NIM | Peran | Tanggung Jawab |
|---|---|---|---|---|
| 1 | [Moh.Nabil Syahputra] | [F55124079] | Backend Developer | Logika di balik aplikasi: pemuatan model `.pkl`, praproses dan prediksi segmen, perhitungan evaluasi (agregasi jendela 30 hari, deteksi outlier, Silhouette, DBI), validasi isian dan berkas CSV, serta penanganan pesan kesalahan (`Menu/prediksi.py` dan `Menu/evaluasi.py` bagian fungsi) |
| 2 | [Salsabila] | [F55124044] | Frontend Developer | Antarmuka Streamlit: `app.py` (sidebar dan navigasi), Beranda, tata letak Simulator Prediksi dan Evaluasi Model (kartu, tabel, grafik Plotly), tema warna, dan tampilan pesan |
| 3 | [Lionel Geraldo Mansa] | [F55124064] | Data Analyst / Machine Learning Engineer | Pemahaman dan pembersihan data, pemilihan fitur RFM, pemodelan di RapidMiner, replikasi dan pelatihan K-Means di Python (`training.py`), penentuan nilai k, serta evaluasi Silhouette, DBI, dan WCSS |