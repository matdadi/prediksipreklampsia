#Definisikan library
from matplotlib.backends.backend_agg import RendererAgg

import matplotlib
import pandas as pd
import streamlit as st
import seaborn as sns
from c45 import C45
import joblib
import io
import numpy as np
from sklearn.model_selection import train_test_split
import time
from streamlit_option_menu import option_menu


#parameter
text_option = '- Pilih -'

st.set_page_config(layout="wide")

matplotlib.use('agg')
_lock=RendererAgg.lock

with st.sidebar:
    choose = option_menu("Menu", ["Preparation", "Prediction"],
                         icons=['archive', 'tv'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "#1d3985", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#a8bbf0"},
    }
    )

# if choose=="Menu":
#     st.title('Prediksi Preklampsia dengan C4.5')

if choose=="Preparation":
    #judul app
    st.title('Preparation')

    st.write('')

    #set grid
    sns.set_style('darkgrid')
    row1_space1, row1, row1_space3 = st.columns(
        (.1, 2, .1)
    )

    with row1, _lock:
        uploaded_file = st.file_uploader("Choose a datashet")
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            # st.write(bytes_data)

            # To convert to a string based IO:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            # st.write(stringio)

            # To read file as string:
            string_data = stringio.read()
            # st.write(string_data)

            # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_csv(uploaded_file, sep=';')
            df = pd.DataFrame(dataframe)

            st.dataframe(df.head(5))

            # clean = clean.dropna()
            clean = df.copy()
            clean.rename(columns = {'kelas':'Decision'}, inplace = True)
            clean = clean.reset_index(drop=True)


            with st.spinner("Training dataset..."):
                time.sleep(5)
            st.success("Done!")
            clf = C45(attrNames=clean.columns[:-1])
            X_train, X_test, y_train, y_test = train_test_split(clean[clean.columns[:-1]], clean['Decision'], test_size=0.2, shuffle=True)
            pred = clf.fit(X_train, y_train)
            joblib.dump(clf, 'model_c45.sav')
            C45(attrNames=['n', 'pe', 'peb'])
            st.write(f'Training Accuracy: {clf.score(X_train, y_train)}')
            st.write(f'Testing Accuracy: {clf.score(X_test, y_test)}')

if choose=="Prediction":
    #judul app
    st.title('Prediksi Preklampsia dengan C4.5')

    st.write('')

    #set grid
    sns.set_style('darkgrid')
    row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns(
        (.1, 1, .1, 1, .1)
    )

    with row1_1, _lock:
        st.subheader('Masukkan data pasien')
        #Pilih Perguruan Tinggi
        pendidikan = st.selectbox(
        'Pendidikan Terakhir?',
        (text_option,'SD', 'SMP', 'SMA', 'PERGURUAN TINGGI'))
        if pendidikan!=text_option:
            if pendidikan=='SD': pendidikan=1
            if pendidikan=='SMP': pendidikan=2
            if pendidikan=='SMA': pendidikan=3
            if pendidikan=='PERGURUAN TINGGI': pendidikan=4
            st.write('You selected:', pendidikan)
            

        #Pilih Pekerjaan
        pekerjaan = st.selectbox(
        'Pekerjaan?',
        (text_option,'ACOUNTING',
                    'IRT',
                    'SWASTA',
                    'PNS',
                    'WIRASWASTA',
                    'SECURITY',
                    'GURU',
                    'MAHASISWA',
                    'DOKTER',
                    'TNI',
                    'POLISI',
                    'SWASTA',
                    'APOTEKER',
                    'SATPOL PP',
                    'HONORER',
                    'BURUH'))
        if pekerjaan!=text_option:
            if pekerjaan=='ACOUNTING': pekerjaan=1
            if pekerjaan=='IRT': pekerjaan=2
            if pekerjaan=='SWASTA': pekerjaan=3
            if pekerjaan=='PNS': pekerjaan=4
            if pekerjaan=='WIRASWASTA': pekerjaan=5
            if pekerjaan=='SECURITY': pekerjaan=6
            if pekerjaan=='GURU': pekerjaan=7
            if pekerjaan=='MAHASISWA': pekerjaan=8
            if pekerjaan=='DOKTER': pekerjaan=9
            if pekerjaan=='TNI': pekerjaan=10
            if pekerjaan=='POLISI': pekerjaan=11
            if pekerjaan=='SWASTA': pekerjaan=12
            if pekerjaan=='APOTEKER': pekerjaan=13
            if pekerjaan=='SATPOL PP': pekerjaan=14
            if pekerjaan=='HONORER': pekerjaan=15
            if pekerjaan=='BURUH': pekerjaan=16
            st.write('You selected:', pekerjaan)

        #Ketik umur
        umur = st.number_input('Umur', value=0, max_value=60)
        if umur !=0:
            st.write('The current number is ', umur)

        # #Ketik usia kehamilan
        # usia_kehamilan = st.number_input('Usia kehamilan', value=0, max_value=42)
        # if usia_kehamilan !=0:
        #     st.write('The current number is ', usia_kehamilan)

        # #Ketik tekanan darah sistol
        # BPS = st.number_input('Sistol', value=50, max_value=400)
        # if BPS !=0:
        #     st.write('The current number is ', BPS)
        
        # #Ketik tekanan darah diastol
        # BPD = st.number_input('Diastol', value=40, max_value=200)
        # if BPD !=0:
        #     st.write('The current number is ', BPD)

        # #Ketik berat badan
        # BB = st.number_input('Berat badan', value=0.)
        # if BB !=0:
        #     st.write('The current number is ', BB)

        #Pilih Jenis Kehamilan
        jenis = st.selectbox(
        'Jenis kehamilan?',
        (text_option,'Normal', 'Kembar'))
        if jenis!=text_option:
            if jenis=='Normal': jenis=1
            else: 2
            st.write('You selected:', jenis)

        #Pilih frekuensi kehamilan
        frekuensi = st.number_input('Kehamilan ke-', value=1, max_value=10)
        if frekuensi !=0:
            st.write('The current number is ', frekuensi)

        #Pilih paritas
        paritas = st.number_input('Paritas', value=0, max_value=10)
        if paritas !=0:
            st.write('The current number is ', paritas)

        #Pilih riwayat aborsi
        abortus = st.number_input('Riwayat aborsi', value=0, max_value=10)
        if abortus !=0:
            st.write('The current number is ', abortus)

        #Pilih riwayat melahirkan
        riwayat_melahirkan = st.selectbox(
        'Riwayat Melahirkan?',
        (text_option,'Belum', 'Caesar', 'Normal', 'Ekstraksi Vakum'))
        if riwayat_melahirkan!=text_option:
            if riwayat_melahirkan=='Belum': riwayat_melahirkan=0
            if riwayat_melahirkan=='Caesar': riwayat_melahirkan=1
            if riwayat_melahirkan=='Normal': riwayat_melahirkan=2
            if riwayat_melahirkan=='Ekstraksi Vakum': riwayat_melahirkan=3
            st.write('You selected:', riwayat_melahirkan)

        #Pilih riwayat penyakit
        st.markdown('Riwayat penyakit:')
        IsAsma = st.checkbox('Asthma')
        IsBatuEmpedu = st.checkbox('Batu Empedu')
        IsDiabetes = st.checkbox('Diabetes Melitus')
        IsHepatitis = st.checkbox('Hepatitis')
        IsHipertensi = st.checkbox('Hipertensi')
        IsJantung = st.checkbox('Jantung')
        IsPe = st.checkbox('Pe')
        IsPeb = st.checkbox('Peb')
        IsTBC = st.checkbox('TBC')
        IsMagh = st.checkbox('Magh')
        if IsAsma==True: IsAsma=1
        else: IsAsma=0

        if IsBatuEmpedu==True: IsBatuEmpedu=1
        else: IsBatuEmpedu=0

        if IsDiabetes==True: IsDiabetes=1
        else: IsDiabetes=0

        if IsHepatitis==True: IsHepatitis=1
        else: IsHepatitis=0

        if IsHipertensi==True: IsHipertensi=1
        else: IsHipertensi=0

        if IsJantung==True: IsJantung=1
        else: IsJantung=0

        if IsPe==True: IsPe=1
        else: IsPe=0

        if IsPeb==True: IsPeb=1
        else: IsPeb=0

        if IsTBC==True: IsTBC=1
        else: IsTBC=0

        if IsMagh==True: IsMagh=1
        else: IsMagh=0
        #Pilih riwayat proteinuira
        proteinuria = st.selectbox(
        'Proteinuira?',
        (text_option,'Negatif', 'Trace', 'Positif 1', 'Positif 2', 'Positif 3', 'Positif 4'))
        if proteinuria!=text_option:
            if proteinuria=='Negatif': proteinuria=0
            if proteinuria=='Trace': proteinuria=1
            if proteinuria=='Positif 1': proteinuria=2
            if proteinuria=='Positif 2': proteinuria=3
            if proteinuria=='Positif 3': proteinuria=4
            if proteinuria=='Positif 4': proteinuria=5
            st.write('You selected:', proteinuria)
        
        if st.button('Prediksi'):
            st.write('Executed')
            model = joblib.load('model_c45.sav')
        
            X_predict = [[pendidikan, pekerjaan, jenis, frekuensi, paritas,
                    abortus, riwayat_melahirkan, IsTBC, IsAsma,
                    IsDiabetes, IsHipertensi, IsHepatitis, 
                    IsJantung, IsPe, IsPeb, IsBatuEmpedu,
                    IsMagh, proteinuria]]

            result = model.predict(X_predict)
            if result[0]==0: pred='negatif'
            if result[0]==1: pred='pe'
            if result[0]==2: pred='peb'
            # st.write('Test:', X_predict)
            st.write('Hasil:', pred)