# Sistem Prediksi Penyakit Hewan Ternak

## Deskripsi Proyek

Sistem ini bertujuan untuk membantu peternak atau ahli kesehatan hewan dalam mendiagnosis penyakit pada hewan ternak berdasarkan gejala yang terlihat. Aplikasi ini memanfaatkan model machine learning dan deep learning untuk melakukan prediksi penyakit berdasarkan data yang diberikan, seperti jenis hewan, umur, suhu tubuh, dan gejala-gejala yang muncul pada hewan. Proyek ini mengintegrasikan beberapa model klasifikasi, termasuk **Random Forest**, **XGBoost**, dan **Feedforward Neural Network**, untuk mencapai tingkat akurasi yang optimal dalam melakukan prediksi.

### Tujuan Pengembangan
1. Menyediakan alat bantu diagnosis penyakit hewan ternak yang cepat dan akurat.
2. Meningkatkan efisiensi dalam pengambilan keputusan terkait kesehatan hewan ternak.
3. Menggunakan machine learning untuk mempelajari pola gejala penyakit dan prediksi penyakit berdasarkan data yang dikumpulkan.

## Langkah Instalasi

### Prasyarat
Sebelum memulai, pastikan bahwa Anda telah menginstal Python 3.7 atau lebih baru. Anda juga memerlukan `pip` untuk mengelola dependensi.

### 1. Instalasi Dependensi
Clone repositori ini ke dalam direktori lokal Anda:

```bash```
git clone https://github.com/username/repository.git
cd repository

Instalasi dependensi Python yang diperlukan:

```bash```
Copy code :
pip install -r requirements.txt
File requirements.txt harus mencakup, di antaranya:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
keras
tensorflow
streamlit
joblib

### 2. Menjalankan Aplikasi Web
Untuk menjalankan aplikasi web menggunakan Streamlit, jalankan perintah berikut:

```bash```
Copy code :
streamlit run app.py
Aplikasi web akan terbuka di browser Anda. Anda dapat mengaksesnya melalui URL yang tertera di terminal.

Deskripsi Model
Model yang Digunakan
Random Forest Classifier
Model ini digunakan untuk klasifikasi penyakit hewan ternak berdasarkan gejala yang terlihat. Random Forest merupakan algoritma ensemble yang memanfaatkan banyak pohon keputusan untuk meningkatkan akurasi prediksi.

XGBoost Classifier
XGBoost adalah algoritma gradient boosting yang terkenal dengan performanya dalam berbagai kompetisi machine learning. Model ini digunakan untuk meningkatkan akurasi dan mengatasi masalah overfitting yang mungkin terjadi pada model lain.

Feedforward Neural Network (FFNN)
Neural Network digunakan untuk membangun model klasifikasi yang lebih kompleks dengan memanfaatkan banyak lapisan neuron. Model ini dilatih menggunakan dataset gejala penyakit pada hewan ternak.

Analisis Performa Model
Model yang digunakan diuji menggunakan dataset yang mencakup gejala penyakit dan klasifikasi penyakit hewan ternak. Akurasi dari setiap model dievaluasi dengan menggunakan metrik accuracy, precision, recall, dan f1-score.

Berikut adalah hasil evaluasi model yang digunakan:

Random Forest Classifier
Akurasi: 95.34%
Classification Report:
Precision: 0.95
Recall: 0.96
F1-Score: 0.95

XGBoost Classifier
Akurasi: 96.52%
Classification Report:
Precision: 0.96
Recall: 0.96
F1-Score: 0.96

Feedforward Neural Network (FFNN)
Akurasi: 97.25%
Classification Report:
Precision: 0.97
Recall: 0.97
F1-Score: 0.97

Kesimpulan
Dari hasil yang diperoleh, dapat disimpulkan bahwa Feedforward Neural Network menunjukkan performa yang sedikit lebih baik daripada Random Forest dan XGBoost dalam hal akurasi, meskipun semua model memberikan hasil yang sangat baik dalam klasifikasi penyakit hewan ternak. XGBoost juga menawarkan hasil yang cukup kompetitif, dengan waktu pelatihan yang lebih singkat dibandingkan dengan model neural network.

Kontribusi
Jika Anda ingin berkontribusi pada proyek ini, silakan fork repositori ini dan buat pull request. Setiap kontribusi dalam bentuk perbaikan bug, penambahan fitur, atau peningkatan dokumentasi sangat diterima.
