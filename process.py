import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import io
import base64
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = ['Calories', 'Total Fat (g)', 'Sodium (mg)', 'Carbs (g)', 'Protein (g)']
DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), 'data', 'FastFoodNutritionMenuV2.csv')


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_default_data():
    return pd.read_csv(DEFAULT_DATASET)

def load_uploaded_data(file_stream):
    return pd.read_csv(file_stream)

def load_manual_data(rows):
    df = pd.DataFrame(rows)
    df['Company'] = 'Manual Input'
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_data(df):
    col_map = {}
    for col in df.columns:
        # Bersihkan nama kolom: buat huruf kecil, hapus enter (\n), dan hapus spasi ujung
        c = col.lower().replace('\n', ' ').strip()
        
        if 'total fat' in c:
            col_map[col] = 'Total Fat (g)'
        elif 'sodium' in c:
            col_map[col] = 'Sodium (mg)'
        elif 'carb' in c:  # Mencari kata 'carb', lebih aman untuk "Carbs\n(g)"
            col_map[col] = 'Carbs (g)'
        elif 'protein' in c: # Mencari kata 'protein'
            col_map[col] = 'Protein (g)'
        elif 'calories' in c and 'fat' not in c:
            col_map[col] = 'Calories'
            
    df = df.rename(columns=col_map)
    
    # Memastikan kolom yang dibutuhkan ada di dataframe
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
            
    # Membersihkan data dari karakter non-numerik (misal: ada simbol < atau koma)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), 
            errors='coerce'
        ).fillna(0)
    
    return df


# ─────────────────────────────────────────────
# 3. ELBOW METHOD
# ─────────────────────────────────────────────

def run_elbow(X_scaled, max_k=10):
    """Hitung inertia untuk K=1 sampai max_k"""
    k_list = list(range(1, min(max_k + 1, len(X_scaled))))
    inertia = []
    for k in k_list:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
    return k_list, inertia


# ─────────────────────────────────────────────
# 4. SILHOUETTE SCORE PER K
# ─────────────────────────────────────────────

def run_silhouette_scores(X_scaled, max_k=10):
    """Hitung silhouette score untuk K=2 sampai max_k"""
    # max K tidak boleh >= jumlah sampel
    max_k_safe = min(max_k, len(X_scaled) - 1)
    if max_k_safe < 2:
        return [], []

    k_list = list(range(2, max_k_safe + 1))
    scores = []
    for k in k_list:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(round(score, 4))
    return k_list, scores


# ─────────────────────────────────────────────
# 5. K-MEANS UTAMA
# ─────────────────────────────────────────────

def run_kmeans(df_clean, n_clusters):
    X = df_clean[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df_out = df_clean.copy()
    df_out['Cluster'] = labels

    sil = None
    if n_clusters > 1 and len(df_clean) > n_clusters:
        try:
            sil = round(silhouette_score(X_scaled, labels), 4)
        except Exception:
            pass

    return df_out, X_scaled, sil


# ─────────────────────────────────────────────
# 6. RINGKASAN CLUSTER
# ─────────────────────────────────────────────

def get_cluster_summary(df_clustered, n_clusters):
    label_sets = {
        2: ['🟢 Low Calorie', '🔴 High Calorie'],
        3: ['🟢 Low Calorie', '🟡 Moderate', '🔴 High Calorie'],
        4: ['🟢 Very Low', '🔵 Low-Moderate', '🟡 Moderate-High', '🔴 High Calorie'],
        5: ['🟢 Very Low', '🔵 Low', '🟡 Moderate', '🟠 High', '🔴 Very High'],
        6: ['🟢 Very Low', '🔵 Low', '🟡 Moderate', '🟠 Mod-High', '🔴 High', '⚫ Very High'],
    }
    labels_name = label_sets.get(n_clusters, [f'Cluster {i}' for i in range(n_clusters)])

    means  = df_clustered.groupby('Cluster')[FEATURE_COLS].mean()
    counts = df_clustered.groupby('Cluster').size()

    # Urutkan berdasarkan rata-rata kalori (rendah ke tinggi)
    sorted_ids = means['Calories'].sort_values().index.tolist()
    label_map  = {cid: labels_name[i] for i, cid in enumerate(sorted_ids)}

    summaries = []
    for cid in range(n_clusters):
        m = means.loc[cid]
        summaries.append({
            'cluster_id'  : int(cid),
            'label'       : label_map.get(cid, f'Cluster {cid}'),
            'count'       : int(counts.get(cid, 0)),
            'avg_calories': round(float(m['Calories']), 1),
            'avg_fat'     : round(float(m['Total Fat (g)']), 1),
            'avg_sodium'  : round(float(m['Sodium (mg)']), 1),
            'avg_carbs'   : round(float(m['Carbs (g)']), 1),
            'avg_protein' : round(float(m['Protein (g)']), 1),
        })

    summaries.sort(key=lambda x: x['avg_calories'])
    return summaries


# ─────────────────────────────────────────────
# 7. HELPER PLOT → BASE64
# ─────────────────────────────────────────────

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ─────────────────────────────────────────────
# 8. PLOT ELBOW
# ─────────────────────────────────────────────

def generate_elbow_plot(k_range, inertia, optimal_k=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    ax.plot(k_range, inertia,
            marker='o', color='#f97316',
            linewidth=2.5, markersize=8,
            markerfacecolor='#fff',
            markeredgecolor='#f97316',
            markeredgewidth=2,
            zorder=3)

    # Highlight K optimal jika ada
    if optimal_k and optimal_k in k_range:
        idx = k_range.index(optimal_k)
        ax.scatter([optimal_k], [inertia[idx]],
                   color='#22d3ee', s=130, zorder=5,
                   label=f'K optimal = {optimal_k}')
        ax.axvline(x=optimal_k, color='#22d3ee',
                   linestyle='--', linewidth=1.5, alpha=0.5)
        ax.legend(facecolor='#1e293b', edgecolor='#475569',
                  labelcolor='#cbd5e1', fontsize=10)

    ax.set_xlabel('Jumlah Cluster (K)', color='#94a3b8', fontsize=11)
    ax.set_ylabel('Inertia (WCSS)', color='#94a3b8', fontsize=11)
    ax.set_title('Metode Elbow — Penentuan K Optimal',
                 color='#f1f5f9', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='#64748b')
    ax.set_xticks(k_range)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, color='#334155', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 9. PLOT SILHOUETTE
# ─────────────────────────────────────────────

def generate_silhouette_plot(k_range, scores, best_k=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    ax.plot(k_range, scores,
            marker='s', color='#22d3ee',
            linewidth=2.5, markersize=8,
            markerfacecolor='#fff',
            markeredgecolor='#22d3ee',
            markeredgewidth=2,
            zorder=3)

    # Highlight K terbaik
    if best_k and best_k in k_range:
        idx = k_range.index(best_k)
        ax.scatter([best_k], [scores[idx]],
                   color='#f97316', s=130, zorder=5,
                   label=f'K terbaik = {best_k} (Score={scores[idx]:.4f})')
        ax.axvline(x=best_k, color='#f97316',
                   linestyle='--', linewidth=1.5, alpha=0.5)
        ax.legend(facecolor='#1e293b', edgecolor='#475569',
                  labelcolor='#cbd5e1', fontsize=10)

    ax.set_xlabel('Jumlah Cluster (K)', color='#94a3b8', fontsize=11)
    ax.set_ylabel('Silhouette Score', color='#94a3b8', fontsize=11)
    ax.set_title('Silhouette Score — Kualitas Cluster per K',
                 color='#f1f5f9', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='#64748b')
    ax.set_xticks(k_range)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, color='#334155', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 10. PLOT SCATTER
# ─────────────────────────────────────────────

def generate_scatter_plot(df_clustered, n_clusters):
    COLORS = ['#22c55e', '#f97316', '#3b82f6', '#a855f7', '#ef4444', '#eab308']

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    for cid in range(n_clusters):
        mask = df_clustered['Cluster'] == cid
        ax.scatter(
            df_clustered.loc[mask, 'Calories'],
            df_clustered.loc[mask, 'Total Fat (g)'],
            c=COLORS[cid % len(COLORS)],
            label=f'Cluster {cid}',
            alpha=0.75,
            edgecolors='none',
            s=55
        )

    ax.set_xlabel('Calories (kcal)', color='#94a3b8', fontsize=11)
    ax.set_ylabel('Total Fat (g)', color='#94a3b8', fontsize=11)
    ax.set_title('Scatter Plot — Calories vs Total Fat',
                 color='#f1f5f9', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='#64748b')
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, color='#334155', linestyle='--', alpha=0.5)
    ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#cbd5e1')

    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 11. FUNGSI UTAMA — dipanggil dari app.py
# ─────────────────────────────────────────────

def run_full_pipeline(source='default', file_stream=None, manual_rows=None, n_clusters=3):
    # ── Load ──
    if source == 'default':
        df = load_default_data()
    elif source == 'upload':
        df = load_uploaded_data(file_stream)
    elif source == 'manual':
        df = load_manual_data(manual_rows)
    else:
        raise ValueError(f'Source tidak dikenal: {source}')

    # ── Preprocess ──
    df_clean = preprocess_data(df)

    # Validasi: jumlah data bersih harus >= n_clusters
    if len(df_clean) < n_clusters:
        raise ValueError(
            f'Data yang valid hanya {len(df_clean)} baris, '
            f'tidak cukup untuk {n_clusters} cluster. '
            f'Tambah data atau kurangi nilai K.'
        )

    # ── Scale untuk Elbow & Silhouette ──
    X_for_eval = StandardScaler().fit_transform(df_clean[FEATURE_COLS].values)

    # ── Elbow ──
    max_k = min(10, len(df_clean) - 1)  # max K tidak boleh >= jumlah data
    k_elbow, inertia = run_elbow(X_for_eval, max_k=max_k)

    # Tentukan K optimal dari elbow (titik penurunan terbesar)
    optimal_k = None
    if len(inertia) >= 2:
        deltas = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
        opt_idx = deltas.index(max(deltas))
        optimal_k = k_elbow[opt_idx + 1]

    # ── Silhouette per K ──
    k_sil, sil_scores = run_silhouette_scores(X_for_eval, max_k=max_k)

    best_k_sil = None
    if sil_scores:
        best_k_sil = k_sil[sil_scores.index(max(sil_scores))]

    # ── K-Means dengan K pilihan user ──
    df_clustered, X_scaled, sil_final = run_kmeans(df_clean, n_clusters)

    # ── Ringkasan cluster ──
    summaries = get_cluster_summary(df_clustered, n_clusters)

    # ── Generate semua plot ──
    elbow_img      = generate_elbow_plot(k_elbow, inertia, optimal_k=optimal_k)
    silhouette_img = generate_silhouette_plot(k_sil, sil_scores, best_k=best_k_sil) if k_sil else None
    scatter_img    = generate_scatter_plot(df_clustered, n_clusters)

    # ── Tabel ──
    table_cols = ['Company', 'Item'] + FEATURE_COLS + ['Cluster']
    avail = [c for c in table_cols if c in df_clustered.columns]
    table_data = df_clustered[avail].copy()
    table_data['Cluster'] = table_data['Cluster'].astype(int)
    table_records = table_data.to_dict(orient='records')

    # ── CSV download ──
    buf = io.StringIO()
    table_data.to_csv(buf, index=False)
    csv_data = buf.getvalue()

    return {
        'total_items'      : len(df_clustered),
        'n_clusters'       : n_clusters,
        'silhouette_score' : sil_final,
        'optimal_k_elbow'  : optimal_k,
        'best_k_silhouette': best_k_sil,
        'summaries'        : summaries,
        'elbow_img'        : elbow_img,
        'silhouette_img'   : silhouette_img,
        'scatter_img'      : scatter_img,
        'table_records'    : table_records,
        'csv_data'         : csv_data,
        'feature_cols'     : FEATURE_COLS,
    }


# ─────────────────────────────────────────────
# 12. JALANKAN LANGSUNG DARI TERMINAL
#     $ python process.py
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    # ── Ubah backend matplotlib ke TkAgg agar gambar bisa tampil di layar ──
    matplotlib.use('TkAgg')

    # Path folder static/img (relatif dari lokasi process.py)
    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    IMG_DIR   = os.path.join(BASE_DIR, 'static', 'img')
    os.makedirs(IMG_DIR, exist_ok=True)

    N_CLUSTERS = 3   # Ubah angka ini jika mau coba K lain

    print("=" * 55)
    print("  SEGMENTASI FAST FOOD — K-Means Clustering")
    print("  python process.py")
    print("=" * 55)

    # ── 1. Load & Preprocess ──────────────────────────────
    print("\n[1/5] Memuat dataset...")
    df_raw   = load_default_data()
    df_clean = preprocess_data(df_raw)
    print(f"      Dataset asli  : {len(df_raw)} baris, {len(df_raw.columns)} kolom")
    print(f"      Setelah bersih: {len(df_clean)} baris")
    print(f"      Fitur digunakan: {FEATURE_COLS}")

    # ── 2. Statistik deskriptif ───────────────────────────
    print("\n[2/5] Statistik deskriptif fitur nutrisi:")
    stats = df_clean[FEATURE_COLS].describe().round(2)
    print(stats.to_string())

    # ── 3. Elbow & Silhouette ─────────────────────────────
    print("\n[3/5] Menghitung Elbow Method & Silhouette Score...")
    X_eval   = StandardScaler().fit_transform(df_clean[FEATURE_COLS].values)
    max_k    = min(10, len(df_clean) - 1)

    k_elbow, inertia = run_elbow(X_eval, max_k=max_k)
    k_sil,   sil_sc  = run_silhouette_scores(X_eval, max_k=max_k)

    # Cari K optimal
    deltas    = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
    optimal_k = k_elbow[deltas.index(max(deltas)) + 1]
    best_k    = k_sil[sil_sc.index(max(sil_sc))] if sil_sc else None

    print("\n      Elbow — Inertia per K:")
    for k, v in zip(k_elbow, inertia):
        bar = '█' * int(v / max(inertia) * 30)
        print(f"      K={k:2d} | {bar:<30} {v:,.1f}")

    print(f"\n       K optimal (Elbow)      : K = {optimal_k}")

    if sil_sc:
        print("\n      Silhouette Score per K:")
        for k, s in zip(k_sil, sil_sc):
            bar = '█' * int(s * 50)
            print(f"      K={k:2d} | {bar:<25} {s:.4f}")
        print(f"\n       K terbaik (Silhouette) : K = {best_k} (score={max(sil_sc):.4f})")

    # ── 4. K-Means dengan K pilihan ───────────────────────
    print(f"\n[4/5] Menjalankan K-Means dengan K = {N_CLUSTERS}...")
    df_clustered, X_scaled, sil_final = run_kmeans(df_clean, N_CLUSTERS)
    summaries = get_cluster_summary(df_clustered, N_CLUSTERS)

    print(f"\n      Silhouette Score (K={N_CLUSTERS}): {sil_final}")
    print("\n      ── Ringkasan Cluster ──────────────────────────────")
    print(f"      {'Cluster':<10} {'Label':<25} {'Items':>6} {'Kalori':>8} {'Fat':>6} {'Sodium':>8} {'Carbs':>7} {'Protein':>8}")
    print("      " + "-" * 80)
    for s in summaries:
        print(
            f"      {s['cluster_id']:<10} "
            f"{s['label']:<25} "
            f"{s['count']:>6} "
            f"{s['avg_calories']:>7.1f} "
            f"{s['avg_fat']:>6.1f} "
            f"{s['avg_sodium']:>8.1f} "
            f"{s['avg_carbs']:>7.1f} "
            f"{s['avg_protein']:>8.1f}"
        )

    # Contoh 5 produk per cluster
    print("\n      ── Contoh Produk per Cluster ──────────────────────")
    for cid in range(N_CLUSTERS):
        subset = df_clustered[df_clustered['Cluster'] == cid][['Company','Item','Calories']].head(5)
        lbl = next((s['label'] for s in summaries if s['cluster_id'] == cid), f'Cluster {cid}')
        print(f"\n      {lbl}:")
        for _, row in subset.iterrows():
            print(f"        • {row['Company']:15} | {row['Item'][:35]:35} | {row['Calories']:.0f} kcal")

    # ── 5. Simpan & tampilkan gambar (style putih bersih seperti modul) ──
    print(f"\n[5/5] Membuat & menyimpan grafik ke {IMG_DIR} ...")

    plt.style.use('default')  # reset ke style putih bersih matplotlib

    # ── Plot Elbow ──
    fig_elbow, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_elbow, inertia, marker='o', color='steelblue', linewidth=2, markersize=7)
    if optimal_k in k_elbow:
        idx = k_elbow.index(optimal_k)
        ax.scatter([optimal_k], [inertia[idx]], color='red', s=100, zorder=5,
                   label=f'K optimal = {optimal_k}')
        ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.legend(fontsize=10)
    ax.set_xlabel('Jumlah Cluster (K)', fontsize=11)
    ax.set_ylabel('Inertia (WCSS)', fontsize=11)
    ax.set_title('Metode Elbow untuk Menentukan K Optimal', fontsize=13, fontweight='bold')
    ax.set_xticks(k_elbow)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    path_elbow = os.path.join(IMG_DIR, 'elbow_plot.png')
    fig_elbow.savefig(path_elbow, dpi=150, bbox_inches='tight')
    print(f"       Tersimpan: {path_elbow}")

    # ── Plot Silhouette ──
    if k_sil:
        fig_sil, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(k_sil, sil_sc, marker='o', color='steelblue', linewidth=2, markersize=7)
        if best_k in k_sil:
            idx2 = k_sil.index(best_k)
            ax2.scatter([best_k], [sil_sc[idx2]], color='red', s=100, zorder=5,
                        label=f'K terbaik = {best_k} ({sil_sc[idx2]:.4f})')
            ax2.axvline(x=best_k, color='red', linestyle='--', linewidth=1.2, alpha=0.6)
            ax2.legend(fontsize=10)
        ax2.set_xlabel('Jumlah Cluster (K)', fontsize=11)
        ax2.set_ylabel('Silhouette Score', fontsize=11)
        ax2.set_title('Metode Silhouette Score', fontsize=13, fontweight='bold')
        ax2.set_xticks(k_sil)
        ax2.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        path_sil = os.path.join(IMG_DIR, 'silhouette_plot.png')
        fig_sil.savefig(path_sil, dpi=150, bbox_inches='tight')
        print(f"       Tersimpan: {path_sil}")

    # ── Plot Scatter ──
    SCATTER_COLORS = ['green', 'orange', 'steelblue', 'purple', 'red', 'goldenrod']
    fig_sc, ax3 = plt.subplots(figsize=(9, 5))
    for cid in range(N_CLUSTERS):
        mask = df_clustered['Cluster'] == cid
        lbl  = next((s['label'] for s in summaries if s['cluster_id'] == cid), f'Cluster {cid}')
        ax3.scatter(df_clustered.loc[mask, 'Calories'],
                    df_clustered.loc[mask, 'Total Fat (g)'],
                    c=SCATTER_COLORS[cid % len(SCATTER_COLORS)],
                    label=lbl, alpha=0.7, edgecolors='none', s=55)
    ax3.set_xlabel('Calories (kcal)', fontsize=11)
    ax3.set_ylabel('Total Fat (g)', fontsize=11)
    ax3.set_title(f'Hasil Clustering Fast Food (K={N_CLUSTERS})', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    path_sc = os.path.join(IMG_DIR, 'cluster_plot.png')
    fig_sc.savefig(path_sc, dpi=150, bbox_inches='tight')
    print(f"       Tersimpan: {path_sc}")

    print("\n" + "=" * 55)
    print("  Semua grafik tersimpan di folder static/img/")
    print("  Menampilkan grafik... (tutup jendela untuk selesai)")
    print("=" * 55)

    plt.show()