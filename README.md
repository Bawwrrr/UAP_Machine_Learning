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
```bash
git clone https://github.com/username/repository.git
cd repository
```

Instalasi dependensi Python yang diperlukan:

```bash
pip install -r requirements.txt
```

File `requirements.txt` harus mencakup, di antaranya:

-   pandas
-   numpy
-   matplotlib
-   seaborn
-   scikit-learn
-   xgboost
-   keras
-   tensorflow
-   streamlit
-   joblib

### 2. Menjalankan Aplikasi Web

Untuk menjalankan aplikasi web menggunakan Streamlit, jalankan perintah berikut:

```bash
streamlit run app.py
```

Aplikasi web akan terbuka di browser Anda. Anda dapat mengaksesnya melalui URL yang tertera di terminal.

## Deskripsi Model

### Model yang Digunakan

1.  **Random Forest Classifier**  
    Model ini digunakan untuk klasifikasi penyakit hewan ternak berdasarkan gejala yang terlihat. Random Forest merupakan algoritma ensemble yang memanfaatkan banyak pohon keputusan untuk meningkatkan akurasi prediksi.
    
2.  **XGBoost Classifier**  
    XGBoost adalah algoritma gradient boosting yang terkenal dengan performanya dalam berbagai kompetisi machine learning. Model ini digunakan untuk meningkatkan akurasi dan mengatasi masalah overfitting yang mungkin terjadi pada model lain.
    
3.  **Feedforward Neural Network (FFNN)**  
    Neural Network digunakan untuk membangun model klasifikasi yang lebih kompleks dengan memanfaatkan banyak lapisan neuron. Model ini dilatih menggunakan dataset gejala penyakit pada hewan ternak.
    

### Analisis Performa Model

Model yang digunakan diuji menggunakan dataset yang mencakup gejala penyakit dan klasifikasi penyakit hewan ternak. Akurasi dari setiap model dievaluasi dengan menggunakan metrik **accuracy**, **precision**, **recall**, dan **f1-score**.

Berikut adalah hasil evaluasi model yang digunakan:

#### Random Forest Classifier

-   **Akurasi:** 80.54%
-   **Classification Report:**
    -   Precision: 0.76
    -   Recall: 0.76
    -   F1-Score: 0.76

#### XGBoost Classifier

-   **Akurasi:** 81.60%
-   **Classification Report:**
    -   Precision: 0.77
    -   Recall: 0.77
    -   F1-Score: 0.77

#### Feedforward Neural Network (FFNN)

-   **Akurasi:** 83.61%
-   **Classification Report:**
    -   Precision: 0.80
    -   Recall: 0.80
    -   F1-Score: 0.75

## Hasil dan Analisis

### Grafik Confusion Matrix

Untuk memberikan gambaran lebih jelas mengenai performa model, berikut adalah grafik confusion matrix untuk model **Random Forest** dan **XGBoost**.

#### Random Forest Confusion Matrix

![Confusion Matrix RF](https://github.com/Bawwrrr/UAP_Machine_Learning/blob/main/Hasill/RF.png)

#### XGBoost Confusion Matrix

![Confusion Matrix XGBoost](https://github.com/Bawwrrr/UAP_Machine_Learning/blob/main/Hasill/XGB.png)

### Grafik Akurasi dan Kerugian Feedforward Neural Network

Grafik ini menunjukkan perubahan **akurasi** dan **kerugian** selama pelatihan model Feedforward Neural Network.

![Accuracy and Loss](https://github.com/Bawwrrr/UAP_Machine_Learning/blob/main/Hasill/FF.png)

## Kesimpulan

Dari hasil yang diperoleh, dapat disimpulkan bahwa **Feedforward Neural Network** menunjukkan performa yang sedikit lebih baik daripada **Random Forest** dan **XGBoost** dalam hal akurasi, meskipun semua model memberikan hasil yang sangat baik dalam klasifikasi penyakit hewan ternak. **XGBoost** juga menawarkan hasil yang cukup kompetitif, dengan waktu pelatihan yang lebih singkat dibandingkan dengan model neural network.

## Kontribusi

Jika Anda ingin berkontribusi pada proyek ini, silakan fork repositori ini dan buat pull request. Setiap kontribusi dalam bentuk perbaikan bug, penambahan fitur, atau peningkatan dokumentasi sangat diterima.

### Penjelasan README:

- **Deskripsi Proyek** menguraikan latar belakang dan tujuan pengembangan proyek.
- **Langkah Instalasi** memberikan petunjuk bagaimana menginstal dan menjalankan aplikasi web.
- **Deskripsi Model** menjelaskan model yang digunakan, seperti Random Forest, XGBoost, dan Feedforward Neural Network (FFNN), serta cara mengevaluasi performa model.
- **Hasil dan Analisis** memberikan informasi mengenai hasil evaluasi model beserta visualisasi confusion matrix dan grafik akurasi/loss untuk model FFNN.

Jika ada hal lain yang perlu ditambahkan atau disesuaikan, beri tahu saya!

```
