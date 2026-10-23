# WiFi CSI Human Activity Recognition (HAR) - Deep Learning

Proyek ini merupakan implementasi *pipeline Deep Learning* untuk klasifikasi aktivitas manusia (*Human Activity Recognition*) menggunakan data seri waktu dari sinyal WiFi Channel State Information (CSI). Proyek ini dikembangkan untuk penyelesaian Track A: HAR & Gesture Recognition.

## Deskripsi Model
Sistem ini membandingkan dua arsitektur jaringan saraf:
1. **Baseline 1D-CNN**: Model *Convolutional Neural Network* 1 dimensi standar sebagai tolok ukur.
2. **Proposed ResNet**: Model arsitektur *Residual Network* 1D usulan yang dilengkapi dengan *Residual Blocks*, *Batch Normalization*, dan *Dropout* untuk ekstraksi fitur yang lebih stabil dan mencegah *overfitting*.

## Dataset
Dataset yang digunakan adalah **UT-HAR** yang mendeteksi 7 kelas aktivitas manusia:
1. Berbaring (Lie Down)
2. Jatuh (Fall)
3. Berjalan (Walk)
4. Memungut (Pickup)
5. Berlari (Run)
6. Duduk (Sit Down)
7. Berdiri (Stand Up)

## Struktur Direktori
- `models/`: Berisi arsitektur model (`baseline_cnn.py` dan `proposed_resnet.py`).
- `utils/`: Berisi fungsi *preprocessing* (Butterworth Low-pass Filter & Z-score Normalization) dan pembacaan *dataloader* PyTorch.
- `main.py`: Skrip utama untuk melatih model (*training*), evaluasi (*validation*), dan visualisasi *Confusion Matrix*.
- `requirements.txt`: Daftar dependensi pustaka Python.

## Cara Menjalankan
1. Instal seluruh dependensi yang dibutuhkan:
   pip install -r requirements.txt
2. Pastikan file dataset berformat `.npy` berada pada direktori yang tepat atau sesuaikan path pada `main.py`.
3. Jalankan skrip utama:
   python main.py
