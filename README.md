---
title: Ravenstack Churn
emoji: 💻
colorFrom: red
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: predict customer churn
---

# 🔮 RavenStack — Customer Churn Intelligence Platform

Aplikasi analisis & prediksi risiko churn pelanggan SaaS B2B, dibangun end-to-end
dari data engineering, feature engineering, model evaluation, hingga dashboard
interaktif berbasis Streamlit.

> ⚠️ Ini proyek portofolio menggunakan **dataset sintetis**. Lihat bagian
> [Keterbatasan Model & Dataset](#-keterbatasan-model--dataset) sebelum menarik
> kesimpulan bisnis dari angka-angka di sini.

---

## 📊 Dataset

**RavenStack Synthetic SaaS Dataset (Multi-Table)**
🔗 Kaggle: https://www.kaggle.com/datasets/rivalytics/saas-subscription-and-churn-analytics-dataset

| Tabel | Baris | Deskripsi |
|---|---|---|
| `accounts` | 500 | 1 baris = 1 akun pelanggan |
| `subscriptions` | 5.000 | Riwayat langganan per akun |
| `feature_usage` | 25.000 | Event penggunaan fitur |
| `support_tickets` | 2.000 | Tiket dukungan pelanggan |
| `churn_events` | 600 | Kejadian churn/reaktivasi |

Kelima tabel ini diagregasi menjadi **83 fitur per akun** (feature engineering
di `src/feature_engineering.py`), menghasilkan `data/processed/ravenstack_features_for_modeling.csv`
— 500 baris, 1 baris per akun, siap untuk modeling.

**Kredit dataset:** River @ Rivalytics ([blog](https://rivalytics.medium.com)).
Lisensi MIT-like, fully synthetic, tanpa PII.

---

## 🧠 Model

| | |
|---|---|
| Algoritma final | Random Forest (Anti-Overfit, `max_depth=10`, `class_weight=balanced`) |
| Jumlah fitur | 83 (seluruh fitur hasil agregasi, bukan subset) |
| Threshold keputusan | 0.33 |
| Recall (test set) | 0.8636 |
| Precision (test set) | 0.3393 |
| F1 | 0.4872 |
| PR-AUC | 0.3805 |
| ROC-AUC | 0.6597 |

**Kenapa threshold rendah & precision "terlihat" rendah itu disengaja:**
biaya kehilangan pelanggan (false negative) jauh lebih mahal dibanding biaya
follow-up CS ke pelanggan yang ternyata aman (false positive). Model dioptimalkan
untuk **recall tinggi**, bukan akurasi keseluruhan.

Alternatif model (XGBoost Optimized) juga tersedia di `models/` untuk perbandingan.

---

## 🖥️ Fitur Aplikasi

- **Input Manual** — masukkan profil pelanggan sendiri untuk melihat skor risiko.
- **Contoh Profil** — 3 contoh pelanggan **sungguhan** dari test set (risiko
  tertinggi, borderline, dan terendah menurut model) — bukan skenario karangan.
- **Dashboard ringkasan** — total pelanggan, churn rate, MRR at risk, dsb.
- **Business Impact** — estimasi kerugian MRR/ARR per akun jika pelanggan
  tersebut benar-benar churn (dihitung terpisah dari skor risiko, murni untuk
  bantu prioritisasi tim CS — lihat catatan di bawah).

---

## ⚠️ Keterbatasan Model & Dataset

Bagian ini sengaja ditulis eksplisit karena kejujuran soal keterbatasan model
adalah bagian dari proses evaluasi yang benar, bukan kegagalan proyek.

### 1. Sinyal fitur terhadap target churn tergolong lemah
Korelasi Pearson tertinggi di antara **seluruh 83 fitur** terhadap target
churn hanya ~0.12 (`days_since_last_usage`). Fitur bisnis intuitif seperti
`avg_satisfaction_score` (r≈0.008) dan `escalation_rate` (r≈0.06) nyaris tidak
berkorelasi dengan churn di dataset ini. Beberapa bahkan berlawanan arah
dengan intuisi bisnis umum (mis. `error_rate` berkorelasi negatif dengan churn).

Praktis: **belum diverifikasi** apakah ini karena ukuran sampel (500 akun)
terlalu kecil, atau karena mekanisme generator sintetis menetapkan `churn_flag`
secara sebagian besar independen dari fitur usage/satisfaction/subscription.
Kandidat langkah lanjutan: generate ulang dataset dengan N lebih besar dan
membandingkan ulang kekuatan korelasinya.

### 2. Skor probabilitas tidak dikalibrasi secara agresif ke ekstrem
Pada test set (data yang tidak dilihat model saat training), probabilitas
prediksi berkisar **0.17–0.54** — model jarang sekali sangat yakin (mendekati
0% atau 100%) untuk data baru, meski pada data training rentangnya jauh lebih
lebar (0.10–0.79). Ini indikasi model masih agak konservatif/underconfident
saat generalisasi ke data baru, konsisten dengan sinyal fitur yang lemah di atas.

### 3. MRR **tidak** dijadikan pengali risiko oleh model
Kartu "Business Impact" (estimasi kerugian MRR/ARR) dihitung **setelah** dan
**terpisah** dari skor probabilitas churn — murni untuk membantu tim CS
memprioritaskan akun mana yang ditangani lebih dulu berdasarkan besar
kerugian potensial, bukan bagian dari perhitungan model itu sendiri. Bahkan,
korelasi `total_mrr` terhadap churn di dataset ini justru lemah dan negatif
(r≈-0.04), sehingga MRR besar tidak secara otomatis menaikkan skor risiko.

Kalau ada penjelasan naratif seperti "pelanggan besar sedang mengevaluasi
kompetitor" di sekitar suatu skor, itu adalah **interpretasi bisnis** untuk
membantu tim CS membaca angka — bukan logika internal model, karena
Random Forest tidak bernalar secara semantik seperti itu.

---

## 🚀 Menjalankan Secara Lokal

```bash
git clone <repo-url>
cd ravenstack-churn
git lfs pull          # penting: model .pkl disimpan via Git LFS
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Struktur Proyek

```
├── app.py                          # Aplikasi Streamlit
├── data/
│   ├── raw/                        # 5 tabel mentah dari Kaggle
│   └── processed/                  # Hasil feature engineering (83 fitur, 500 baris)
├── models/                         # Model terlatih (.pkl) + metadata
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_evaluation_part1.ipynb
│   ├── 03_modeling_comparison.ipynb
│   ├── 04_xgboost_optimized.ipynb
│   ├── 05_deployment.ipynb
│   └── debug_model.ipynb           # Investigasi kompresi probabilitas train vs test
├── reports/                        # Visualisasi EDA, evaluasi model, SHAP
├── src/
│   └── feature_engineering.py      # Agregasi 5 tabel → 83 fitur per akun
└── requirements.txt
```

---

## 🛠️ Tech Stack

Python · Streamlit · scikit-learn · XGBoost · SHAP · pandas · Docker

---

## 👤 Kredit

- **Dataset:** RavenStack Synthetic SaaS Dataset oleh River @ Rivalytics —
  [Kaggle](https://www.kaggle.com/datasets/rivalytics/saas-subscription-and-churn-analytics-dataset) ·
  [Blog](https://rivalytics.medium.com)
- **Model & aplikasi:** Pandu Bashir Alamin

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference