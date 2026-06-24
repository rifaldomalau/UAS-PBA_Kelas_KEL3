import streamlit as st
import joblib
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Config halaman Streamlit
st.set_page_config(page_title="ABSA & NER Dashboard - Kelp3", layout="wide")

# 1. LOAD MODEL-MODEL TERBAIK KELP3 (DENGAN PATH SINKRON LUAR FOLDER)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_proyek_a_models():
    # Merajut jalur folder secara absolut dan aman di sistem operasi Windows
    model_path = os.path.join(BASE_DIR, 'models', 'Kelp3_best_multilabel_model.pkl')
    vectorizer_path = os.path.join(BASE_DIR, 'models', 'Kelp3_tfidf_vectorizer.pkl')
    
    model_ml = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model_ml, vectorizer

@st.cache_resource
def load_proyek_b_models():
    # Merajut jalur absolut ke folder model NER hasil rename sesuai instruksi dosen
    ner_model_path = os.path.join(BASE_DIR, 'models', 'Kelp3_best_ner_model')
    
    # Detektor pengaman tambahan: memunculkan pesan instruksi jika nama folder belum di-rename
    if not os.path.exists(ner_model_path):
        st.error(f"❌ Folder model NER tidak ditemukan di: {ner_model_path}\n\n"
                 f"Pastikan Kelp3 sudah mengubah nama folder 'Kelp3_best_indobert_ner' "
                 f"menjadi 'Kelp3_best_ner_model' di dalam folder models")
        st.stop()
        
    tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
    model_ner = AutoModelForTokenClassification.from_pretrained(ner_model_path)
    
    # Deteksi GPU otomatis (Akselerasi RTX 5060 Ti)
    device = 0 if torch.cuda.is_available() else -1
    
    ner_pipe = pipeline("token-classification", model=model_ner, tokenizer=tokenizer, aggregation_strategy="simple", device=device)
    return ner_pipe

# Eksekusi loading model (Semua fungsi didefinisikan di atas sebelum dipanggil di sini)
with st.spinner("📦 Sedang memuat model-model terbaik Kelp3 ke dalam VRAM..."):
    model_ml, tfidf_vectorizer = load_proyek_a_models()
    ner_pipeline = load_proyek_b_models()

# skema label resmi proyek multilabel
list_label_ml = [
    'PRODUCT_POSITIVE', 'PRODUCT_NEGATIVE', 'PRODUCT_NEUTRAL', 
    'PRICE_POSITIVE', 'PRICE_NEGATIVE', 'PRICE_NEUTRAL',
    'PLACE_POSITIVE', 'PLACE_NEGATIVE', 'PLACE_NEUTRAL',
    'PROMOTION_POSITIVE', 'PROMOTION_NEGATIVE', 'PROMOTION_NEUTRAL', 
    'OUT_OF_TOPIC'
]

# DESAIN INTERFACE (UI) APLIKASI WEB
st.title("🔑 Dashboard Analisis Sentimen Berbasis Aspek (ABSA) & NER")
st.subheader("🤖 Hasil Karya Kelompok 3 - Sistem Deteksi Review Otomatis")
st.markdown("---")

st.markdown("### 📝 Masukkan Teks Review Pelanggan di Sini:")
user_input = st.text_area(
    label="Tuliskan ulasan toko/produk secara bebas untuk diuji oleh AI:",
    value="Puas belanja disini harganya pas di kantong dan pelayanan ayam gepreknya ramah banget!",
    height=100
)

if st.button("🚀 Jalankan Analisis AI", type="primary"):
    if user_input.strip() == "":
        st.warning("⚠️ Mohon isi teks review terlebih dahulu sebelum menekan tombol!")
    else:
        # Layout 2 kolom kiri-kanan
        col1, col2 = st.columns(2)
        
        # PROYEK A: PREDIKSI MULTILABEL TEXT CLASSIFICATION
        with col1:
            st.markdown("### 📊 Proyek A: Multi-label Aspect Classification")
            st.caption("Mendeteksi komponen aspek & sentimen global di dalam kalimat.")
            
            input_clean = user_input.lower().strip()
            input_tfidf = tfidf_vectorizer.transform([input_clean])
            
            pred_ml = model_ml.predict(input_tfidf)[0]
            active_labels = [list_label_ml[i] for i, val in enumerate(pred_ml) if val == 1]
            
            if len(active_labels) > 0:
                st.success(f"🎯 Terdeteksi **{len(active_labels)}** Kategori Aspek-Sentimen:")
                for lbl in active_labels:
                    if "POSITIVE" in lbl:
                        st.markdown(f"🍏 `{lbl}`")
                    elif "NEGATIVE" in lbl:
                        st.markdown(f"🍎 `{lbl}`")
                    else:
                        st.markdown(f"🪙 `{lbl}`")
            else:
                st.info("🪙 `OUT_OF_TOPIC` (Tidak ada aspek utama terdeteksi)")

        # PROYEK B: PREDIKSI NAMED ENTITY RECOGNITION (NER)
        with col2:
            st.markdown("### 🧬 Proyek B: Named Entity Recognition (NER)")
            st.caption("Menyorot frasa kata spesifik yang merujuk pada entitas aspek tertentu.")
            
            ner_results = ner_pipeline(user_input)
            
            if len(ner_results) > 0:
                st.info(f"🔍 Terdeteksi **{len(ner_results)}** Frasa Entitas Khusus:")
                
                entities_data = []
                for ent in ner_results:
                    # Trik Pengaman: Hapus simbol '##' bawaan subword BERT agar teks terbaca natural
                    clean_word = ent['word'].replace("##", "").strip()
                    
                    entities_data.append({
                        "Frasa Kata": clean_word, # <-- Gunakan variabel clean_word yang sudah bersih
                        "Label Entitas": ent['entity_group'],
                        "Skor Keyakinan AI": f"{ent['score']*100:.2f}%"
                    })
                st.table(entities_data)
            else:
                st.write("Tidak ada entitas aspek spesifik (`PRODUCT`, `PRICE`, `PLACE`, `PROMOTION`) yang berhasil disorot.")

st.markdown("---")
st.caption("© 2026 Kelompok 3 - Ujian Akhir Semester Pemrosesan Bahasa Alami.")