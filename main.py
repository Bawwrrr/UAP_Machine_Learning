import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Prediksi Penyakit Hewan Ternak",
    page_icon="🐄",
    layout="wide"
)

# Load the saved models and encoders
@st.cache_resource
def load_models():
    try:
        animal_encoder = joblib.load('./models/animal_encoder.pkl')
        disease_encoder = joblib.load('./models/disease_encoder.pkl')
        scaler = joblib.load('./models/scaler.pkl')
        
        # Load Neural Network model
        try:
            ff_model = tf.keras.models.load_model('./models/feedforward_model.h5')
        except Exception as e:
            st.error(f"Error loading Neural Network model: {str(e)}")
            ff_model = None
            
        return ff_model, animal_encoder, disease_encoder, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

# Define symptoms dictionary with descriptions
symptoms_dict = {
    'blisters on gums': 'Gelembung berisi cairan yang muncul pada gusi hewan',
    'blisters on hooves': 'Gelembung berisi cairan yang muncul pada kuku atau telapak kaki hewan',
    'blisters on mouth': 'Gelembung berisi cairan yang muncul pada mulut hewan',
    'blisters on tongue': 'Gelembung berisi cairan yang muncul pada lidah hewan',
    'chest discomfort': 'Ketidaknyamanan atau rasa sakit di area dada',
    'chills': 'Menggigil atau gemetar yang tidak normal',
    'crackling sound': 'Suara berderak saat bernafas atau bergerak',
    'depression': 'Penurunan aktivitas, lesu, dan kehilangan minat terhadap lingkungan',
    'difficulty walking': 'Kesulitan atau keengganan untuk berjalan',
    'fatigue': 'Kelelahan berlebihan dan kurang energi',
    'lameness': 'Pincang atau ketidaknormalan dalam berjalan',
    'loss of appetite': 'Kehilangan nafsu makan atau menolak makanan',
    'painless lumps': 'Benjolan yang tidak nyeri di berbagai bagian tubuh',
    'shortness of breath': 'Kesulitan bernafas atau nafas pendek',
    'sores on gums': 'Luka terbuka atau lesi pada gusi',
    'sores on hooves': 'Luka terbuka pada kuku atau telapak kaki',
    'sores on mouth': 'Luka terbuka pada mulut',
    'sores on tongue': 'Luka terbuka pada lidah',
    'sweats': 'Berkeringat berlebihan yang tidak normal',
    'swelling in abdomen': 'Pembengkakan pada area perut',
    'swelling in extremities': 'Pembengkakan pada bagian ekstremitas (kaki, ekor)',
    'swelling in limb': 'Pembengkakan pada anggota tubuh',
    'swelling in muscle': 'Pembengkakan pada otot',
    'swelling in neck': 'Pembengkakan pada area leher'
}

# Get symptoms list
symptoms_list = [
    'blisters on gums', 'blisters on hooves', 'blisters on mouth', 
    'blisters on tongue', 'chest discomfort', 'chills', 'crackling sound', 
    'depression', 'difficulty walking', 'fatigue', 'lameness', 
    'loss of appetite', 'painless lumps', 'shortness of breath', 
    'sores on gums', 'sores on hooves', 'sores on mouth', 'sores on tongue', 
    'sweats', 'swelling in abdomen', 'swelling in extremities', 
    'swelling in limb', 'swelling in muscle', 'swelling in neck'
]

def main():
    # Sidebar for model information
    st.sidebar.title("Informasi Model")
    st.sidebar.info("""
    Aplikasi ini menggunakan model Deep Learning:
    - Feedforward Neural Network
    
    Model ini telah dilatih menggunakan dataset penyakit hewan ternak
    dengan berbagai gejala dan karakteristik.
    """)
    
    # Main content
    st.title('Sistem Prediksi Penyakit Hewan Ternak 🏥')
    st.write("""
    Sistem ini membantu memprediksi penyakit hewan ternak berdasarkan karakteristik dan gejala yang terlihat.
    Silakan lengkapi informasi berikut.
    """)
    
    try:
        # Load models
        ff_model, animal_encoder, disease_encoder, scaler = load_models()
        
        if all(model is not None for model in [ff_model, animal_encoder, disease_encoder, scaler]):
            # Create the input interface
            st.subheader('Data Hewan:')
            
            # Create columns for basic information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Animal selection
                animal_options = animal_encoder.classes_
                selected_animal = st.selectbox('Jenis Hewan:', animal_options)
            
            with col2:
                # Age input berdasarkan statistik dataset
                st.write("Statistik Umur:")
                st.write("Min: 1 tahun, Max: 15 tahun")
                age = st.number_input('Umur (tahun):', 
                                    min_value=1.0, 
                                    max_value=15.0, 
                                    value=1.0,  # median dari dataset
                                    step=1.0,
                                    help="Rentang umur: 1-15 tahun, dengan rata-rata 6.77 tahun")
            
            with col3:
                # Temperature input berdasarkan statistik dataset
                st.write("Statistik Suhu:")
                st.write("Min: 100°F, Max: 105°F")
                temperature = st.number_input('Suhu Tubuh (°F):',
                                            min_value=100.0,
                                            max_value=105.0,
                                            value=100.0,  # median dari dataset
                                            step=0.1,
                                            help="Rentang suhu: 100-105°F, dengan rata-rata 102.27°F")
            
            # Symptoms selection with descriptions
            st.subheader('Gejala yang Terlihat:')
            st.write('Pilih gejala-gejala yang terlihat pada hewan (minimal 3):')

            selected_symptoms = st.multiselect(
                'Pilih Gejala:',
                symptoms_list,
                format_func=lambda x: f"{x.replace('_', ' ').title()} - {symptoms_dict[x]}",
                help="Pilih minimal 3 gejala yang terlihat pada hewan"
            )

            # Show selected symptoms count
            total_selected = len(selected_symptoms)
            if total_selected > 0:
                st.write(f"Jumlah gejala yang dipilih: {total_selected}")
                if total_selected < 3:
                    st.warning("⚠️ Mohon pilih minimal 3 gejala")
            
            # Create feature vector
            if st.button('Prediksi Penyakit'):
                if len(selected_symptoms) >= 3:
                    # Show progress
                    progress_text = "Sedang melakukan prediksi..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    # Prepare the input data
                    input_data = pd.DataFrame(columns=['Animal', 'Age', 'Temperature'] + symptoms_list)
                    input_data.loc[0] = 0  # Initialize with zeros
                    
                    # Update progress
                    my_bar.progress(25, text="Memproses data input...")
                    
                    # Encode features
                    input_data['Animal'] = animal_encoder.transform([selected_animal])
                    input_data['Age'] = age
                    input_data['Temperature'] = temperature
                    
                    # Encode symptoms
                    for symptom in selected_symptoms:
                        input_data[symptom] = 1
                    
                    # Update progress
                    my_bar.progress(50, text="Melakukan scaling data...")
                    
                    # Scale the features
                    input_scaled = scaler.transform(input_data)
                    
                    # Update progress
                    my_bar.progress(75, text="Membuat prediksi...")
                    
                    try:
                        # Make prediction
                        prediction = ff_model.predict(input_scaled)
                        predicted_disease = disease_encoder.inverse_transform(
                            np.argmax(prediction, axis=1))[0]
                        
                        # Get prediction probability
                        probability = np.max(prediction) * 100
                        
                        # Update progress
                        my_bar.progress(100, text="Prediksi selesai!")
                        
                        # Display results
                        st.subheader('Hasil Prediksi:')
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.success(f'Penyakit yang diprediksi: {predicted_disease}')
                        
                        with col2:
                            st.info(f'Tingkat keyakinan: {probability:.2f}%')
                        
                        # Display input summary
                        st.subheader('Ringkasan Data Input:')
                        st.write(f"""
                        - Jenis Hewan: {selected_animal}
                        - Umur: {age} tahun
                        - Suhu Tubuh: {temperature}°F
                        - Gejala yang Dipilih: {', '.join(selected_symptoms)}
                        """)
                        
                    except Exception as e:
                        st.error(f"Error dalam melakukan prediksi: {str(e)}")
                    
                else:
                    st.warning('⚠️ Mohon pilih minimal 3 gejala untuk melakukan prediksi.')
        else:
            st.error('Model atau encoder gagal dimuat. Mohon periksa file model dan coba lagi.')
    
    except Exception as e:
        st.error(f'Terjadi kesalahan dalam aplikasi: {str(e)}')
        st.error('Pastikan semua file model (.h5 dan .pkl) berada dalam direktori yang sama dengan aplikasi.')
    
    # Footer
    st.markdown("""---""")
    st.markdown("""
    <div style='text-align: center'>
        <p>Dibuat dengan ❤️ menggunakan Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()