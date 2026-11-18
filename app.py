import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import urllib.parse
import random

st.set_page_config(page_title="TripWise Madura", page_icon="🏝️", layout="wide")

@st.cache_data
def load_data_and_encoders():
    try:
        df = pd.read_excel('dataset_bersih_ui.xlsx')
    except FileNotFoundError:
        df = pd.read_excel('dataset_madura_bersih.xlsx')
    
    review_col = None
    for col in ['ulasan_bersih', 'ulasan', 'text_original', 'text', 'review']:
        if col in df.columns:
            review_col = col
            break
            
    reviews_dict = {}
    if review_col:
        valid_reviews = df.dropna(subset=[review_col])
        reviews_dict = valid_reviews.groupby('id_destinasi')[review_col].apply(
            lambda x: list(x.sample(min(len(x), 3), random_state=42))
        ).to_dict()

    meta_df = df[['id_destinasi', 'destinasi', 'wisata']].drop_duplicates(subset=['id_destinasi'])
    meta_dict = meta_df.set_index('id_destinasi').to_dict('index')
    
    for dest_id in meta_dict.keys():
        meta_dict[dest_id]['reviews'] = reviews_dict.get(dest_id, ["Belum ada ulasan yang tersedia."])
    
    df_model = df[['id_destinasi', 'rekomendasi_cuaca']].dropna()
    
    np.random.seed(42)
    df_model['user_id'] = np.random.randint(1, 301, size=len(df_model))
    
    encoders = {
        'user': LabelEncoder().fit(df_model['user_id']),
        'item': LabelEncoder().fit(df_model['id_destinasi']),
        'season': LabelEncoder().fit(df_model['rekomendasi_cuaca'])
    }
    
    unique_destinations = df_model['id_destinasi'].unique()
    unique_seasons = df_model['rekomendasi_cuaca'].unique()
    
    return df_model, encoders, unique_destinations, unique_seasons, meta_dict

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('neumf_seasonal_model.h5')

with st.spinner('Memuat Engine Rekomendasi NeuMF & Basis Data Ulasan...'):
    df_model, encoders, unique_destinations, unique_seasons, meta_dict = load_data_and_encoders()
    model = load_model()

st.title("🏝️ TripWise Madura")
st.markdown("### Sistem Rekomendasi Destinasi Wisata Cerdas Berbasis *Konteks Musim*")
st.write("Temukan destinasi wisata terbaik di Pulau Madura yang aman dan paling cocok dengan cuaca hari ini.")

col1, col2 = st.columns([1, 2.5])

with col1:
    st.header("⚙️ Rencana Perjalanan")
    st.info("Sistem AI kami akan menyesuaikan rekomendasi agar Anda terhindar dari risiko cuaca buruk.")
    
    # Input Nama
    nama_wisatawan = st.text_input("Nama Anda:", placeholder="Contoh: Rifqi")
    
    def percantik_teks_cuaca(cuaca_asli):
        teks = str(cuaca_asli).lower()
        if 'kemarau' in teks and ('hujan' in teks or 'penghujan' in teks):
            return "🌦️ Aman Sepanjang Tahun (Kemarau & Penghujan)"
        elif 'hujan' in teks or 'penghujan' in teks:
            return "🌧️ Musim Penghujan (Cenderung Basah & Rawan Badai)"
        elif 'kemarau' in teks or 'cerah' in teks:
            return "☀️ Musim Kemarau (Cerah, Terik & Langit Bersih)"
        return f"🌡️ Kondisi Cuaca: {cuaca_asli}"

    season_input = st.selectbox(
        "Bagaimana Prediksi Cuaca Saat Kunjungan Anda?", 
        unique_seasons,
        format_func=percantik_teks_cuaca
    )
    
    top_k = st.slider("Berapa Rekomendasi yang Anda Inginkan?", min_value=3, max_value=10, value=5)
    
    predict_button = st.button("🔍 Temukan Destinasi", use_container_width=True, type="primary")

with col2:
    st.header("🎯 Hasil Rekomendasi")
    
    if predict_button:
        if not nama_wisatawan.strip():
            st.warning("Silakan masukkan nama Anda terlebih dahulu sebelum mencari rekomendasi.")
        else:
            with st.spinner("Menganalisis probabilitas kecocokan menggunakan AI..."):
                try:
                    valid_user_ids = encoders['user'].classes_
                    user_hash = abs(hash(nama_wisatawan.lower().strip()))
                    guest_id = valid_user_ids[user_hash % len(valid_user_ids)]
                    
                    numpy_safe_seed = user_hash % (2**32)
                    
                    user_enc_val = encoders['user'].transform([guest_id])[0]
                    season_enc_val = encoders['season'].transform([season_input])[0]
                    
                    num_items = len(unique_destinations)
                    all_items_encoded = np.arange(num_items)
                    
                    user_array = np.full(num_items, user_enc_val)
                    season_array = np.full(num_items, season_enc_val)
                    predictions = model.predict([user_array, all_items_encoded, season_array], verbose=0).flatten()
                    
                    pool_size = min(10, num_items)
                    
                    np.random.seed(numpy_safe_seed)
                    noise_factor = np.random.normal(0, 0.05, num_items)
                    adjusted_predictions = predictions + noise_factor
                    
                    top_pool_indices = adjusted_predictions.argsort()[-pool_size:][::-1]
                    random_top_k_indices = random.sample(list(top_pool_indices), top_k)
                    top_indices = sorted(random_top_k_indices, key=lambda idx: predictions[idx], reverse=True)
                    
                    top_item_encoded = all_items_encoded[top_indices]
                    top_scores = predictions[top_indices]
                    
                    top_destinations_id = encoders['item'].inverse_transform(top_item_encoded)
                    
                    st.success(f"Halo **{nama_wisatawan.title()}**! Ditemukan **{top_k} destinasi terbaik** pada kondisi:\n**{percantik_teks_cuaca(season_input)}**")
                    
                    for rank, (dest_id, score) in enumerate(zip(top_destinations_id, top_scores), start=1):
                        dest_info = meta_dict.get(dest_id, {'destinasi': f'Destinasi {dest_id}', 'wisata': 'Umum', 'reviews': []})
                        nama_destinasi = dest_info['destinasi']
                        kategori = str(dest_info['wisata']).title()
                        persentase = score * 100
                        
                        query_maps = urllib.parse.quote(f"{nama_destinasi} Madura")
                        url_maps = f"https://www.google.com/maps/search/?api=1&query={query_maps}"
                        
                        with st.container():
                            st.markdown(f"### #{rank} - {nama_destinasi.title()}")
                            col_a, col_b = st.columns([3, 1])
                            
                            with col_a:
                                st.write(f"**Kategori Wisata:** 🏷️ {kategori}")
                                st.write(f"**Tingkat Kecocokan:** 🎯 {persentase:.2f}%")
                                st.markdown(f"[📍 Buka Lokasi di Google Maps]({url_maps})")
                            
                            with col_b:
                                st.progress(float(score))
                            
                            with st.expander("💬 Lihat Ulasan Pengunjung"):
                                ulasan_list = dest_info.get('reviews', [])
                                if not ulasan_list or ulasan_list == ["Belum ada ulasan yang tersedia."]:
                                    st.write("Belum ada ulasan yang tersedia untuk destinasi ini.")
                                else:
                                    for idx, ulasan in enumerate(ulasan_list):
                                        st.info(f'"{ulasan}"')
                                        
                            st.divider()

                except Exception as e:
                    st.error(f"Terjadi kesalahan internal komputasi: {e}")
    else:
        st.info("👈 Masukkan nama Anda, pilih cuaca, lalu klik 'Temukan Destinasi' untuk melihat rekomendasi.")

st.markdown("---")
st.caption("Dikembangkan oleh Mohammad Rifqi Razzan Pratama | MBKM Penelitian 2026 | Universitas Trunojoyo Madura")