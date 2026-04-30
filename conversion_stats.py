import pandas as pd

# Load data
df = pd.read_csv("lead101_realtime_5000.csv")

# ─────────────────────────────────────────────
# 1. Overall Conversion Rate
# ─────────────────────────────────────────────
overall_rate = df["converted"].mean() * 100
total_converted = df["converted"].sum()
total_leads = len(df)

print("=" * 55)
print("1. OVERALL CONVERSION RATE")
print("=" * 55)
print(f"  Total Leads    : {total_leads}")
print(f"  Converted      : {total_converted}")
print(f"  Not Converted  : {total_leads - total_converted}")
print(f"  Conversion Rate: {overall_rate:.2f}%")

# ─────────────────────────────────────────────
# 2. Conversion Rate by Source
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("2. CONVERSION RATE BY SOURCE")
print("=" * 55)
by_source = (
    df.groupby("source")["converted"]
    .agg(Total="count", Converted="sum")
    .assign(Conversion_Rate=lambda x: (x["Converted"] / x["Total"] * 100).round(2))
    .sort_values("Conversion_Rate", ascending=False)
)
print(by_source.to_string())

# ─────────────────────────────────────────────
# 3. Conversion Rate by Course Interest
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("3. CONVERSION RATE BY COURSE INTEREST")
print("=" * 55)
by_course = (
    df.groupby("course_interest")["converted"]
    .agg(Total="count", Converted="sum")
    .assign(Conversion_Rate=lambda x: (x["Converted"] / x["Total"] * 100).round(2))
    .sort_values("Conversion_Rate", ascending=False)
)
print(by_course.to_string())

# ─────────────────────────────────────────────
# 4. Conversion Rate by City
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("4. CONVERSION RATE BY CITY")
print("=" * 55)
by_city = (
    df.groupby("city")["converted"]
    .agg(Total="count", Converted="sum")
    .assign(Conversion_Rate=lambda x: (x["Converted"] / x["Total"] * 100).round(2))
    .sort_values("Conversion_Rate", ascending=False)
)
print(by_city.to_string())

# ─────────────────────────────────────────────
# 5. Converted Leads Only — days_to_convert
# ─────────────────────────────────────────────
converted_df = df[df["converted"] == 1]

print("\n" + "=" * 55)
print("5. CONVERTED LEADS — days_to_convert ANALYSIS")
print("=" * 55)
print(f"  Total Converted Leads: {len(converted_df)}")
print(f"  Average days_to_convert: {converted_df['days_to_convert'].mean():.2f}")
print(f"  Minimum days_to_convert: {converted_df['days_to_convert'].min():.0f}")
print(f"  Maximum days_to_convert: {converted_df['days_to_convert'].max():.0f}")
print(f"  Median  days_to_convert: {converted_df['days_to_convert'].median():.0f}")

print("\n--- Distribution of days_to_convert (Buckets) ---")
bins = [0, 15, 30, 45, 60, 75, 90]
labels = ["0-15d", "16-30d", "31-45d", "46-60d", "61-75d", "76-90d"]
converted_df = converted_df.copy()
converted_df["days_bucket"] = pd.cut(
    converted_df["days_to_convert"], bins=bins, labels=labels
)
bucket_dist = converted_df["days_bucket"].value_counts().sort_index()
for bucket, count in bucket_dist.items():
    bar = "#" * (count // 10)
    print(f"  {bucket:>8}: {count:>4}  {bar}")

print("\n--- days_to_convert by Source (Converted Only) ---")
by_source_conv = (
    converted_df.groupby("source")["days_to_convert"]
    .agg(Count="count", Mean="mean", Min="min", Max="max")
    .round(2)
    .sort_values("Mean")
)
print(by_source_conv.to_string())
