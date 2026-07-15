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