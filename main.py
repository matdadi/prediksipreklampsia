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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
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
        uploaded_file = st.file_uploader("Choose a datasheet")
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

            st.dataframe(df.head(10))

            # clean = clean.dropna()
            clean = df.copy()
            clean.rename(columns = {'kelas':'Decision'}, inplace = True)
            clean['BB'] = clean['BB'].replace(',', '.', regex=True).astype('str').astype('float')
            clean = clean.reset_index(drop=True)

            if st.button('Re-Test'):
                with st.spinner("Training dataset..."):
                    clf = C45(attrNames=clean.columns[:-1])
                    X_train, X_test, y_train, y_test = train_test_split(clean[clean.columns[:-1]], clean['Decision'], test_size=0.2, shuffle=True)
                    C45(attrNames=['n', 'pe', 'peb'])
                    train = clf.fit(X_train, y_train)
                    test = clf.predict(X_test)
                    info_res = "Training accuracy: {:.2f}".format(accuracy_score(y_train, train.y_)*100) + "%, Testing accuracy: {:.2f}".format(accuracy_score(y_test, test)*100)
                    st.warning(f'Split training')
                    st.info(info_res)
                    joblib.dump(clf, 'model_c45.sav')
                    st.error('Model saved')

                    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
                    train_res = []
                    test_res = []
                    st.warning(f'10 Fold cross validation')
                    for train, test in kfold.split(clean):
                        X_train = clean[clean.columns[:-1]].iloc[train]
                        y_train = np.ravel(clean[clean.columns[-1:]].iloc[train])
                        X_test = clean[clean.columns[:-1]].iloc[test]
                        y_test = np.ravel(clean[clean.columns[-1:]].iloc[test])
                        
                        y_trained = clf.fit(X_train, y_train)
                        test = clf.predict(X_test)
                        train_res.append(accuracy_score(y_train, y_trained.y_))
                        test_res.append(accuracy_score(y_test, test))
                        # info_res = "Training accuracy: {:.2f}".format(accuracy_score(y_train, y_trained.y_)*100) + "%, Testing accuracy: {:.2f}".format(accuracy_score(y_test, test)*100)+"%"
                        # st.info(info_res)
                        # st.write('')
                    info_res = "Validated training accuracy: {:.2f}".format(np.array(train_res).mean()*100) + "%, Validated Testing accuracy: {:.2f}".format(np.array(test_res).mean()*100)+"%"
                    # info_res = "Validated Testing accuracy: {:.2f}".format(np.array(test_res).mean()*100)+"%"
                    st.info(info_res)
                st.success("Done!")
                
            else:
                with st.spinner("Training dataset..."):
                    clf = C45(attrNames=clean.columns[:-1])
                    X_train, X_test, y_train, y_test = train_test_split(clean[clean.columns[:-1]], clean['Decision'], test_size=0.2, shuffle=True)
                    C45(attrNames=['n', 'pe', 'peb'])
                    train = clf.fit(X_train, y_train)
                    test = clf.predict(X_test)
                    info_res = "Training accuracy: {:.2f}".format(accuracy_score(y_train, train.y_)*100) + "%, Testing accuracy: {:.2f}".format(accuracy_score(y_test, test)*100)
                    st.warning(f'Split training')
                    st.info(info_res)
                    joblib.dump(clf, 'model_c45.sav')
                    st.error('Model saved')

                    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
                    train_res = []
                    test_res = []
                    st.warning(f'10 Fold cross validation')
                    for train, test in kfold.split(clean):
                        X_train = clean[clean.columns[:-1]].iloc[train]
                        y_train = np.ravel(clean[clean.columns[-1:]].iloc[train])
                        X_test = clean[clean.columns[:-1]].iloc[test]
                        y_test = np.ravel(clean[clean.columns[-1:]].iloc[test])
                        
                        y_trained = clf.fit(X_train, y_train)
                        test = clf.predict(X_test)
                        train_res.append(accuracy_score(y_train, y_trained.y_))
                        test_res.append(accuracy_score(y_test, test))
                        # info_res = "Training accuracy: {:.2f}".format(accuracy_score(y_train, y_trained.y_)*100) + "%, Testing accuracy: {:.2f}".format(accuracy_score(y_test, test)*100)+"%"
                        # st.info(info_res)
                        # st.write('')
                    info_res = "Validated training accuracy: {:.2f}".format(np.array(train_res).mean()*100) + "%, Validated Testing accuracy: {:.2f}".format(np.array(test_res).mean()*100)+"%"
                    # info_res = "Validated Testing accuracy: {:.2f}".format(np.array(test_res).mean()*100)+"%"
                    st.info(info_res)
                st.success("Done!")
               


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
        if pendidikan==text_option: pendidikan=1000
        if pendidikan=='SD': pendidikan=1
        if pendidikan=='SMP': pendidikan=2
        if pendidikan=='SMA': pendidikan=3
        if pendidikan=='PERGURUAN TINGGI': pendidikan=4            

        #Pilih Pekerjaan
        pekerjaan = st.selectbox(
        'Pekerjaan?',
        (text_option,'ACOUNTING',
                    'APOTEKER',
                    'BURUH',
                    'DOKTER',
                    'GURU',
                    'HONORER',
                    'IRT',
                    'MAHASISWA',
                    'PNS',
                    'POLISI',
                    'SATPOL PP',
                    'SECURITY',
                    'SWASTA',
                    'TNI',
                    'WIRASWASTA'))
        
        if pekerjaan==text_option: pekerjaan=1000
        if pekerjaan=='ACOUNTING': pekerjaan=1
        if pekerjaan=='APOTEKER': pekerjaan=2
        if pekerjaan=='BURUH': pekerjaan=3
        if pekerjaan=='DOKTER': pekerjaan=4
        if pekerjaan=='GURU': pekerjaan=5
        if pekerjaan=='HONORER': pekerjaan=6
        if pekerjaan=='IRT': pekerjaan=7
        if pekerjaan=='MAHASISWA': pekerjaan=8
        if pekerjaan=='PNS': pekerjaan=9
        if pekerjaan=='POLISI': pekerjaan=10
        if pekerjaan=='SATPOL PP': pekerjaan=11
        if pekerjaan=='SECURITY': pekerjaan=12
        if pekerjaan=='SWASTA': pekerjaan=13
        if pekerjaan=='TNI': pekerjaan=14
        if pekerjaan=='WIRASWASTA': pekerjaan=15

        #Ketik umur
        umur = st.number_input('Umur', value=0, max_value=60)
        # if umur !=0:
            # st.write('The current number is ', umur)

        #Ketik usia kehamilan
        usia_kehamilan = st.number_input('Usia kehamilan (minggu)', value=0, max_value=50)
        if usia_kehamilan<14 and usia_kehamilan>0: usia_kehamilan=1
        if usia_kehamilan>13 and usia_kehamilan<28: usia_kehamilan=2
        if usia_kehamilan==0: usia_kehamilan=1000
        else: usia_kehamilan=3

        #Ketik tekanan darah sistol dan diastol
        BP=0
        BPS = st.number_input('Sistol', value=0, max_value=400)
        BPD = st.number_input('Diastol', value=0, max_value=200)
        if BPS !=0 and BPD!=0:
            if BPS>120 and BPD>80: BP = 1
        else: BP=1000
        
        # #Ketik berat badan
        BB = st.number_input('Berat badan', value=0.)
        if BB==0.: BB=1000
        

        #Pilih Jenis Kehamilan
        jenis = st.selectbox(
        'Jenis kehamilan?',
        (text_option,'Normal', 'Kembar'))
        if jenis=='Normal': jenis=0
        if jenis=='Kembar': jenis=1
        if jenis==text_option: jenis=1000

        #Pilih frekuensi kehamilan
        frekuensi = st.number_input('Kehamilan ke-', value=1, max_value=10)
        if frekuensi ==0: frekuensi=1000
        #     st.write('The current number is ', frekuensi)

        #Pilih paritas
        paritas = st.number_input('Paritas', value=0, max_value=10)
        # if paritas !=0:
        #     st.write('The current number is ', paritas)

        #Pilih riwayat aborsi
        abortus = st.number_input('Riwayat aborsi', value=0, max_value=10)
        # if abortus !=0:
        #     st.write('The current number is ', abortus)

        #Pilih riwayat melahirkan
        riwayat_melahirkan = st.selectbox(
        'Riwayat Melahirkan?',
        (text_option,'Belum', 'Normal', 'Caesar', 'Ekstraksi Vakum'))
        if riwayat_melahirkan==text_option: riwayat_melahirkan=1000
        if riwayat_melahirkan=='Belum': riwayat_melahirkan=0
        if riwayat_melahirkan=='Normal': riwayat_melahirkan=1
        if riwayat_melahirkan=='Caesar': riwayat_melahirkan=2
        if riwayat_melahirkan=='Ekstraksi Vakum': riwayat_melahirkan=3

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
        if proteinuria==text_option: proteinuria=1000
        if proteinuria=='Negatif': proteinuria=0
        if proteinuria=='Trace': proteinuria=1
        if proteinuria=='Positif 1': proteinuria=2
        if proteinuria=='Positif 2': proteinuria=3
        if proteinuria=='Positif 3': proteinuria=4
        if proteinuria=='Positif 4': proteinuria=5
        
        
        if st.button('Prediksi'):
            # st.write('Executed')
            model = joblib.load('model_c45.sav')
        
            X_predict = [[pendidikan, pekerjaan, umur, usia_kehamilan,
                    BP, BB, jenis, frekuensi, paritas,
                    abortus, riwayat_melahirkan, IsTBC, IsAsma,
                    IsDiabetes, IsHipertensi, IsHepatitis, 
                    IsJantung, IsPe, IsPeb, IsBatuEmpedu,
                    IsMagh, proteinuria]]
            # st.write(X_predict[0])
            if np.array(X_predict[0]).sum()>999:
                st.error('Isi semua form terlebih dahulu.')
            else:
                result = model.predict(X_predict)
                if result[0]==0: pred='negatif'
                if result[0]==1: pred='pe'
                if result[0]==2: pred='peb'
                # st.write('Test:', X_predict)
                info_pred = 'Hasil prediksi: '+ pred.upper()
                st.info(info_pred)
