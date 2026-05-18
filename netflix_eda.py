# =====================================================
# NETFLIX MOVIES & TV SHOWS - EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================
# PROJECT INFORMATION
# =====================================================
# Dataset: Netflix Movies and TV Shows
# Source: Kaggle
# Tools: Python, Pandas, NumPy, Matplotlib, Seaborn
# Objective: Perform exploratory data analysis and extract insights
# =====================================================

# -------------------------------
# IMPORT REQUIRED LIBRARIES
# -------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# VISUALIZATION SETTINGS
# -------------------------------
sns.set_style("whitegrid")

PRIMARY = "#4C78A8"
SECONDARY = "#F58518"
ACCENT = "#54A24B"
DANGER = "#E45756"

os.makedirs("visuals", exist_ok=True)

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("netflix_titles.csv")

# -------------------------------
# DATA UNDERSTANDING
# -------------------------------
print("\n================ DATASET INFO ================\n")
print(df.info())

print("\n================ FIRST 5 RECORDS ================\n")
print(df.head())

print("\n================ MISSING VALUES ================\n")
print(df.isnull().sum())

print("\n================ DUPLICATE RECORDS ================\n")
print(df.duplicated().sum())

# -------------------------------
# DATA CLEANING
# -------------------------------
df['director'] = df['director'].fillna("Unknown")
df['cast'] = df['cast'].fillna("Unknown")
df['country'] = df['country'].fillna("Unknown")

df = df.dropna(subset=['date_added', 'rating', 'duration']).copy()

df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year

df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')

df = df.reset_index(drop=True)

# -------------------------------
# Summary statistics
# -------------------------------
print("\n================ SUMMARY STATISTICS ================\n")

most_common_year = df['release_year'].mode()[0]
oldest_year = df['release_year'].min()
newest_year = df['release_year'].max()

print(f"Most Common Release Year : {most_common_year}")
print(f"Oldest Release Year      : {oldest_year}")
print(f"Newest Release Year      : {newest_year}")

print(f"Total Titles: {len(df)}")
print(f"Total Movies: {(df['type'] == 'Movie').sum()}")
print(f"Total TV Shows: {(df['type'] == 'TV Show').sum()}")

# =====================================================
# VISUALIZATION 1: CONTENT TYPE DISTRIBUTION
# =====================================================
plt.figure(figsize=(6, 6))

content_counts = df['type'].value_counts()

plt.pie(
    content_counts,
    labels=content_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=[PRIMARY, SECONDARY]
)

plt.legend(
    title="Content Type",
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)

plt.title("Distribution of Movies vs TV Shows on Netflix")
plt.tight_layout()

plt.savefig("visuals/type_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# =====================================================
# VISUALIZATION 2: CONTENT GROWTH OVER TIME
# =====================================================
plt.figure(figsize=(10, 5))

trend_data = df.groupby(['year_added', 'type']).size().unstack(fill_value=0)

sns.lineplot(data=trend_data)

plt.title("Netflix Content Growth Over Time")
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.savefig("visuals/content_growth.png", dpi=300)
plt.show()

# =====================================================
# VISUALIZATION 3: RATINGS DISTRIBUTION
# =====================================================
plt.figure(figsize=(10, 5))

ratings_order = df['rating'].value_counts().index

ax = sns.countplot(
    data=df,
    x='rating',
    order=ratings_order,
    color=PRIMARY
)

ax.bar_label(ax.containers[0])

plt.title("Distribution of Netflix Ratings")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig("visuals/ratings_distribution.png", dpi=300)
plt.show()

# =====================================================
# VISUALIZATION 4: TOP 10 COUNTRIES
# =====================================================
plt.figure(figsize=(10, 5))

countries = df['country'].str.split(', ').explode()
top_countries = countries.value_counts().head(10)

ax = plt.barh(top_countries.index, top_countries.values, color=ACCENT)

plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.title("Top 10 Content-Producing Countries")

plt.bar_label(ax, labels=top_countries.values)

plt.tight_layout()
plt.savefig("visuals/top_countries.png", dpi=300, bbox_inches="tight")
plt.show()

# =====================================================
# VISUALIZATION 5: TOP 10 GENRES
# =====================================================
plt.figure(figsize=(10, 6))

genres = df['listed_in'].str.split(', ').explode()
top_genres = genres.value_counts().head(10)

ax = plt.barh(top_genres.index, top_genres.values, color=DANGER)

plt.xlabel("Number of Titles")
plt.ylabel("Genre")
plt.title("Top 10 Netflix Genres")

plt.bar_label(ax, labels=top_genres.values, padding=3)

plt.subplots_adjust(left=0.35)
plt.tight_layout()

plt.savefig("visuals/top_genres.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------
# EXPORT CLEANED DATASET
# -------------------------------
df.to_csv("netflix_cleaned_dataset.csv", index=False)

print("\n================ COMPLETED SUCCESSFULLY ================")
print("All visualizations saved in 'visuals/' folder")
print("Cleaned dataset exported as 'netflix_cleaned_dataset.csv'")