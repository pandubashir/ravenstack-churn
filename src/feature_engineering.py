"""
RavenStack - Customer Churn Prediction
Feature Engineering Pipeline
======================================
Menggabungkan 5 tabel menjadi satu dataset siap modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: LOAD DATA")
print("=" * 60)

DATA_PATH = "/mnt/user-data/uploads/"

accounts    = pd.read_csv(DATA_PATH + "ravenstack_accounts.csv")
subs        = pd.read_csv(DATA_PATH + "ravenstack_subscriptions.csv")
feature_usage = pd.read_csv(DATA_PATH + "ravenstack_feature_usage.csv")
tickets     = pd.read_csv(DATA_PATH + "ravenstack_support_tickets.csv")
churn_events = pd.read_csv(DATA_PATH + "ravenstack_churn_events.csv")

# Parse tanggal
accounts["signup_date"]       = pd.to_datetime(accounts["signup_date"])
subs["start_date"]            = pd.to_datetime(subs["start_date"])
subs["end_date"]              = pd.to_datetime(subs["end_date"])
feature_usage["usage_date"]   = pd.to_datetime(feature_usage["usage_date"])
tickets["submitted_at"]       = pd.to_datetime(tickets["submitted_at"])
tickets["closed_at"]          = pd.to_datetime(tickets["closed_at"])
churn_events["churn_date"]    = pd.to_datetime(churn_events["churn_date"])

REFERENCE_DATE = pd.Timestamp("2025-01-01")

print(f"accounts      : {accounts.shape}")
print(f"subscriptions : {subs.shape}")
print(f"feature_usage : {feature_usage.shape}")
print(f"support_tickets: {tickets.shape}")
print(f"churn_events  : {churn_events.shape}")
print(f"Reference date: {REFERENCE_DATE.date()}")


# ============================================================
# 2. FEATURE GROUP A — ACCOUNT (Demografis)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: FEATURE GROUP A — ACCOUNT")
print("=" * 60)

feat_account = accounts[["account_id", "industry", "country",
                          "signup_date", "referral_source",
                          "plan_tier", "seats", "is_trial", "churn_flag"]].copy()

# Tenure: lama berlangganan (hari)
feat_account["tenure_days"] = (REFERENCE_DATE - feat_account["signup_date"]).dt.days

# Tenure bucket
feat_account["tenure_bucket"] = pd.cut(
    feat_account["tenure_days"],
    bins=[0, 90, 180, 365, 730, 9999],
    labels=["0-3mo", "3-6mo", "6-12mo", "1-2yr", "2yr+"]
)

# Seats bucket
feat_account["seats_bucket"] = pd.cut(
    feat_account["seats"],
    bins=[0, 5, 20, 50, 9999],
    labels=["micro", "small", "medium", "large"]
)

# Drop kolom raw
feat_account.drop(columns=["signup_date"], inplace=True)

print(f"Features: {feat_account.shape[1]-1} fitur dari {feat_account.shape[0]} akun")
print(feat_account[["tenure_days","tenure_bucket","seats_bucket"]].head(3))


# ============================================================
# 3. FEATURE GROUP B — SUBSCRIPTION
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: FEATURE GROUP B — SUBSCRIPTION")
print("=" * 60)

# Agregasi per account_id
sub_agg = subs.groupby("account_id").agg(
    total_subscriptions       = ("subscription_id", "count"),
    active_subscriptions      = ("churn_flag", lambda x: (x == False).sum()),
    churned_subscriptions     = ("churn_flag", lambda x: (x == True).sum()),
    total_mrr                 = ("mrr_amount", "sum"),
    avg_mrr                   = ("mrr_amount", "mean"),
    max_mrr                   = ("mrr_amount", "max"),
    total_arr                 = ("arr_amount", "sum"),
    num_upgrades              = ("upgrade_flag", "sum"),
    num_downgrades            = ("downgrade_flag", "sum"),
    num_trials                = ("is_trial", "sum"),
    has_auto_renew            = ("auto_renew_flag", "any"),
    pct_annual_billing        = ("billing_frequency", lambda x: (x == "annual").mean()),
    latest_plan_tier          = ("plan_tier", "last"),  # plan paling akhir
    subscription_duration_avg = ("start_date", lambda x: (
        (REFERENCE_DATE - pd.to_datetime(x)).dt.days.mean()
    )),
).reset_index()

# Rasio churn subscription
sub_agg["sub_churn_ratio"] = (
    sub_agg["churned_subscriptions"] / sub_agg["total_subscriptions"]
).fillna(0)

# Upgrade/downgrade pernah terjadi
sub_agg["ever_upgraded"]   = (sub_agg["num_upgrades"] > 0).astype(int)
sub_agg["ever_downgraded"] = (sub_agg["num_downgrades"] > 0).astype(int)

# Net plan movement
sub_agg["net_plan_movement"] = sub_agg["num_upgrades"] - sub_agg["num_downgrades"]

print(f"Features: {sub_agg.shape[1]-1} fitur subscription")
print(sub_agg[["total_mrr","avg_mrr","sub_churn_ratio","net_plan_movement"]].head(3))


# ============================================================
# 4. FEATURE GROUP C — FEATURE USAGE (Engagement)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: FEATURE GROUP C — FEATURE USAGE")
print("=" * 60)

# Join usage dengan subscription untuk dapat account_id
fu_with_account = feature_usage.merge(
    subs[["subscription_id", "account_id"]], on="subscription_id", how="left"
)

# Agregasi per account_id
fu_agg = fu_with_account.groupby("account_id").agg(
    total_usage_events        = ("usage_id", "count"),
    total_usage_count         = ("usage_count", "sum"),
    avg_usage_count           = ("usage_count", "mean"),
    total_usage_duration_secs = ("usage_duration_secs", "sum"),
    avg_usage_duration_secs   = ("usage_duration_secs", "mean"),
    total_error_count         = ("error_count", "sum"),
    avg_error_count           = ("error_count", "mean"),
    unique_features_used      = ("feature_name", "nunique"),
    beta_feature_usage_count  = ("is_beta_feature", "sum"),
    usage_days                = ("usage_date", "nunique"),   # hari unik aktif
    last_usage_date           = ("usage_date", "max"),
).reset_index()

# Recency: berapa hari sejak terakhir pakai fitur
fu_agg["days_since_last_usage"] = (
    REFERENCE_DATE - pd.to_datetime(fu_agg["last_usage_date"])
).dt.days

# Error rate per event
fu_agg["error_rate"] = (
    fu_agg["total_error_count"] / fu_agg["total_usage_events"]
).fillna(0)

# Feature breadth ratio (dari 40 fitur tersedia)
fu_agg["feature_breadth_ratio"] = fu_agg["unique_features_used"] / 40

# Usage density: rata-rata event per hari aktif
fu_agg["usage_density"] = (
    fu_agg["total_usage_events"] / fu_agg["usage_days"]
).fillna(0)

# Recency bucket
fu_agg["recency_bucket"] = pd.cut(
    fu_agg["days_since_last_usage"],
    bins=[-1, 7, 30, 90, 9999],
    labels=["active_7d", "active_30d", "active_90d", "inactive"]
)

fu_agg.drop(columns=["last_usage_date"], inplace=True)

print(f"Features: {fu_agg.shape[1]-1} fitur engagement")
print(fu_agg[["unique_features_used","days_since_last_usage","error_rate","feature_breadth_ratio"]].head(3))


# ============================================================
# 5. FEATURE GROUP D — SUPPORT TICKETS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: FEATURE GROUP D — SUPPORT TICKETS")
print("=" * 60)

tickets_agg = tickets.groupby("account_id").agg(
    total_tickets             = ("ticket_id", "count"),
    avg_resolution_hours      = ("resolution_time_hours", "mean"),
    max_resolution_hours      = ("resolution_time_hours", "max"),
    avg_first_response_mins   = ("first_response_time_minutes", "mean"),
    avg_satisfaction_score    = ("satisfaction_score", "mean"),   # ada nulls → mean otomatis skip
    min_satisfaction_score    = ("satisfaction_score", "min"),
    num_escalated             = ("escalation_flag", "sum"),
    num_urgent_tickets        = ("priority", lambda x: (x == "urgent").sum()),
    num_high_tickets          = ("priority", lambda x: (x == "high").sum()),
    pct_urgent                = ("priority", lambda x: (x == "urgent").mean()),
).reset_index()

# Escalation rate
tickets_agg["escalation_rate"] = (
    tickets_agg["num_escalated"] / tickets_agg["total_tickets"]
).fillna(0)

# Satisfaction flag: pernah dapat score <= 2 (sangat tidak puas)
tickets_agg["has_low_satisfaction"] = (
    tickets_agg["min_satisfaction_score"] <= 2
).astype(int)

# Kompleksitas masalah: banyak ticket urgent+high
tickets_agg["critical_ticket_count"] = (
    tickets_agg["num_urgent_tickets"] + tickets_agg["num_high_tickets"]
)

print(f"Features: {tickets_agg.shape[1]-1} fitur support")
print(tickets_agg[["total_tickets","avg_satisfaction_score","escalation_rate","pct_urgent"]].head(3))


# ============================================================
# 6. FEATURE GROUP E — CHURN EVENTS (sinyal historis)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: FEATURE GROUP E — CHURN EVENTS (sinyal historis)")
print("=" * 60)

churn_agg = churn_events.groupby("account_id").agg(
    churn_event_count          = ("churn_event_id", "count"),
    total_refund_usd           = ("refund_amount_usd", "sum"),
    avg_refund_usd             = ("refund_amount_usd", "mean"),
    had_preceding_upgrade      = ("preceding_upgrade_flag", "any"),
    had_preceding_downgrade    = ("preceding_downgrade_flag", "any"),
    is_reactivation            = ("is_reactivation", "any"),
    last_churn_date            = ("churn_date", "max"),
).reset_index()

# One-hot reason_code (ambil yang paling sering per account)
reason_mode = (churn_events.groupby("account_id")["reason_code"]
               .agg(lambda x: x.mode()[0])
               .reset_index()
               .rename(columns={"reason_code": "main_churn_reason"}))
churn_agg = churn_agg.merge(reason_mode, on="account_id", how="left")

# Days since last churn event
churn_agg["days_since_last_churn"] = (
    REFERENCE_DATE - pd.to_datetime(churn_agg["last_churn_date"])
).dt.days

churn_agg.drop(columns=["last_churn_date"], inplace=True)

print(f"Features: {churn_agg.shape[1]-1} fitur churn historis")
print(churn_agg[["churn_event_count","total_refund_usd","main_churn_reason"]].head(3))


# ============================================================
# 7. MERGE SEMUA TABEL → MASTER DATASET
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: MERGE SEMUA TABEL")
print("=" * 60)

master = feat_account.copy()
master = master.merge(sub_agg,     on="account_id", how="left")
master = master.merge(fu_agg,      on="account_id", how="left")
master = master.merge(tickets_agg, on="account_id", how="left")
master = master.merge(churn_agg,   on="account_id", how="left")

# Akun tanpa tiket support → isi 0
ticket_cols = [c for c in tickets_agg.columns if c != "account_id"]
master[ticket_cols] = master[ticket_cols].fillna(0)

# Akun tanpa churn event historis
churn_hist_cols = ["churn_event_count","total_refund_usd","avg_refund_usd",
                   "days_since_last_churn"]
master[churn_hist_cols] = master[churn_hist_cols].fillna(0)
master["had_preceding_upgrade"]   = master["had_preceding_upgrade"].fillna(False)
master["had_preceding_downgrade"] = master["had_preceding_downgrade"].fillna(False)
master["is_reactivation"]         = master["is_reactivation"].fillna(False)
master["main_churn_reason"]       = master["main_churn_reason"].fillna("none")

print(f"Master dataset: {master.shape}")
print(f"Null per kolom:\n{master.isnull().sum()[master.isnull().sum()>0]}")


# ============================================================
# 8. ENCODING & FINALISASI
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: ENCODING FITUR KATEGORIKAL")
print("=" * 60)

# Label encode ordinal
plan_order = {"Basic": 0, "Pro": 1, "Enterprise": 2}
master["plan_tier_encoded"]        = master["plan_tier"].map(plan_order)
master["latest_plan_tier_encoded"] = master["latest_plan_tier"].map(plan_order)

tenure_order = {"0-3mo": 0, "3-6mo": 1, "6-12mo": 2, "1-2yr": 3, "2yr+": 4}
master["tenure_bucket_encoded"] = master["tenure_bucket"].map(tenure_order)

seats_order = {"micro": 0, "small": 1, "medium": 2, "large": 3}
master["seats_bucket_encoded"] = master["seats_bucket"].map(seats_order)

recency_order = {"active_7d": 0, "active_30d": 1, "active_90d": 2, "inactive": 3}
master["recency_bucket_encoded"] = master["recency_bucket"].map(recency_order)

# One-hot encode nominal
nominal_cols = ["industry", "country", "referral_source", "main_churn_reason"]
master = pd.get_dummies(master, columns=nominal_cols, drop_first=False, dtype=int)

# Boolean → int
bool_cols = master.select_dtypes(include="bool").columns.tolist()
master[bool_cols] = master[bool_cols].astype(int)

# Target variable
master["target"] = master["churn_flag"].astype(int)

# Drop kolom yang tidak diperlukan untuk modeling
drop_cols = ["account_id", "account_name", "churn_flag",
             "plan_tier", "latest_plan_tier",
             "tenure_bucket", "seats_bucket", "recency_bucket"]
master_model = master.drop(columns=[c for c in drop_cols if c in master.columns])

print(f"Dataset akhir: {master_model.shape}")
print(f"Target distribusi:\n{master_model['target'].value_counts()}")
print(f"Churn rate: {master_model['target'].mean():.2%}")


# ============================================================
# 9. SIMPAN OUTPUT
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: SIMPAN OUTPUT")
print("=" * 60)

# Dataset lengkap (dengan nama kolom readable)
master.to_csv("/home/claude/ravenstack_master_raw.csv", index=False)

# Dataset siap modeling (sudah encoded)
master_model.to_csv("/home/claude/ravenstack_features_for_modeling.csv", index=False)

print("Saved: ravenstack_master_raw.csv")
print("Saved: ravenstack_features_for_modeling.csv")

# ============================================================
# 10. RINGKASAN FITUR
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN FITUR YANG DIBUAT")
print("=" * 60)

feature_summary = {
    "A. Account (Demografis)": [
        "tenure_days", "tenure_bucket_encoded", "seats", "seats_bucket_encoded",
        "is_trial", "plan_tier_encoded", "industry_*", "country_*", "referral_source_*"
    ],
    "B. Subscription": [
        "total_subscriptions", "active_subscriptions", "churned_subscriptions",
        "total_mrr", "avg_mrr", "max_mrr", "total_arr",
        "num_upgrades", "num_downgrades", "sub_churn_ratio",
        "ever_upgraded", "ever_downgraded", "net_plan_movement",
        "pct_annual_billing", "has_auto_renew", "latest_plan_tier_encoded"
    ],
    "C. Feature Usage (Engagement)": [
        "total_usage_events", "total_usage_count", "avg_usage_count",
        "total_usage_duration_secs", "avg_usage_duration_secs",
        "total_error_count", "avg_error_count", "error_rate",
        "unique_features_used", "feature_breadth_ratio",
        "beta_feature_usage_count", "usage_days",
        "days_since_last_usage", "recency_bucket_encoded", "usage_density"
    ],
    "D. Support Tickets": [
        "total_tickets", "avg_resolution_hours", "max_resolution_hours",
        "avg_first_response_mins", "avg_satisfaction_score", "min_satisfaction_score",
        "num_escalated", "escalation_rate", "pct_urgent",
        "has_low_satisfaction", "critical_ticket_count"
    ],
    "E. Churn Historis": [
        "churn_event_count", "total_refund_usd", "avg_refund_usd",
        "had_preceding_upgrade", "had_preceding_downgrade",
        "is_reactivation", "days_since_last_churn", "main_churn_reason_*"
    ],
}

total = 0
for group, feats in feature_summary.items():
    print(f"\n{group} ({len(feats)} fitur):")
    for f in feats:
        print(f"   - {f}")
    total += len(feats)

print(f"\nTotal fitur (sebelum one-hot expand): ~{total}")
print(f"Total kolom final dataset: {master_model.shape[1]-1} fitur + 1 target")
print("\n✅ Feature Engineering selesai!")
