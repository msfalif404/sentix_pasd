#----------------------------------Importing Library----------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from google_play_scraper import reviews
from wordcloud import WordCloud

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
#--------------------------------End of importing library-----------------------------------

#------------------------------------Interactive-------------------------------------------------

def urlParse(url):
    # Mengambil query parameter "id" tanpa menggunakan urllib.parse
    query_string = url.split("?")[-1]  # Memisahkan query string dari URL
    query_params = query_string.split("&")  # Memisahkan query parameter
    id_value = ""
    for param in query_params:
        key, value = param.split("=")  # Memisahkan kunci dan nilai
        if key == "id":
            id_value = value
            break

    # Menampilkan nilai dari query parameter "id"
    return id_value


st.set_page_config(page_title = "5entiX")
st.title("Google Play Store Review Analysis")

if "cv_state" not in st.session_state:
    st.session_state["cv_state"] = 0

if "classifier_state" not in st.session_state:
    st.session_state["classifier_state"] = 0

if "end_state" not in st.session_state:
    st.session_state["end_state"] = False

with st.form(key = 'form1'):
    # Menerima input dari pengguna
    app_link = st.text_input("Masukkan link aplikasi:")
    submit_button = st.form_submit_button(label="Check")

    if submit_button:
        # Check if the input is a Play Store link
        playstore_link = "play.google.com/store/apps"
        if playstore_link not in app_link:
            st.error("Input tidak valid. Harap masukkan tautan Play Store yang valid.")
        else:
            app_link = urlParse(app_link)
            
            #--------------------------------Scraping The Application Reviews---------------------------
            result, continuation_token = reviews(
                app_link,
                lang = 'en',
                country = 'us',
                count = 5000,
            )

            #creating dataframe from scraping result
            df = pd.DataFrame(result)
            df = df[["content","score"]]
            #-----------------------------End of Scraping The Application Reviews-------------------------
            
            #Menampilkan 5 baris pertama
            st.header('Dataframe Awal:')
            st.write(df.head())

            #-------------------------Membuat diagram batang untuk kolom skor------------------------------

            # Menghitung jumlah data untuk setiap skor
            score_counts = df['score'].value_counts().sort_index()

            # Menampilkan diagram batang menggunakan Streamlit
            st.header("Menampilkan Grafik Skor")
            st.bar_chart(data = score_counts)
            #-------------------------End Of Membuat diagram batang untuk kolom skor----------------------

            #Mengklasifikasikan sentimen berdasarkan rating
            df['score'] = df['score'].apply(lambda x: 1 if x > 3 else -1)

            #-------------------------Membuat Word Clouds Positif---------------------------------------

            # Menggabungkan teks dari kolom "content" menjadi satu string
            text = " ".join(df[df['score'] == 1]['content'])

            # Membuat WordCloud
            st.set_option('deprecation.showPyplotGlobalUse', False)
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

            # Menampilkan WordCloud menggunakan Streamlit
            fig, ax = plt.subplots()
            st.header("Menampilkan WordCloud Positif")
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot()
            #-------------------------End of Membuat Word Clouds Positif-----------------------------------

            #-------------------------Membuat Word Clouds Negatif------------------------------------------

            # Menggabungkan teks dari kolom "content" menjadi satu string
            text = " ".join(df[df['score'] == -1]['content'])

            # Membuat WordCloud
            st.set_option('deprecation.showPyplotGlobalUse', False)
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

            # Menampilkan WordCloud menggunakan Streamlit
            fig, ax = plt.subplots()
            st.header("Menampilkan WordCloud Negatif")
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot()
            #-------------------------End of Membuat Word Clouds Negatif-----------------------------------

            #----------------------------Cleaning The Dataset---------------------------------------------
            clean_review = []
            for i in range(len(df)):
                #Menghilangkan semua karakter kecuali huruf
                review = re.sub('[^a-zA-Z]', ' ', df['content'][i])
                
                #Merubah kalimat menjadi huruf kecil
                review = review.lower()

                #Memisahkan kalimat berdasarkan spasi
                review = review.split()
                
                lemmatizer = WordNetLemmatizer()
                all_stopwords = stopwords.words('english')
                all_stopwords.remove('not')
                review = [lemmatizer.lemmatize(word) for word in review if not word in set(all_stopwords)]
                review = ' '.join(review)
                clean_review.append(review)
            
            df_clean_review = pd.DataFrame(clean_review)
            #--------------------------End Of Cleaning The Dataset------------------------------------------

            # Menampilkan 5 baris pertama setelah data dibersihkan
            # st.header("Dataframe Setelah Dibersihkan")
            # st.write(df_clean_review.head())

            #-------------------------Creating The Bag Of Word Model----------------------------------------
            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features = 1500)
            X = cv.fit_transform(clean_review).toarray()
            y = df.iloc[:, -1].values

            st.session_state["cv_state"] = cv
            #------------------End of Creating The Bag Of Word Model----------------------------------------

            #-----------------Splitting The Dataset---------------------------------------------------------
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
            #----------------End Of Slitting The Dataset----------------------------------------------------

            #---------------Training The SVM Model----------------------------------------------------------
            from sklearn.svm import SVC
            classifier = SVC(kernel = 'rbf', random_state = 0)
            classifier.fit(X_train, y_train)
            
            st.session_state["classifier_state"] = classifier
            #--------------End Of Training The SVM Model----------------------------------------------------

            #--------------Making The Confusion Matrix------------------------------------------------------
            from sklearn.metrics import confusion_matrix, accuracy_score
            y_pred = classifier.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            st.header("Akurasi Model")
            st.write(accuracy_score(y_test, y_pred))
            #------------------------------------------------------------------------------------------------

            st.header("Prediksi Sentiment Beberapa Teks")
            text_to_check = ["i hate this game so much, this game is so bad, so slow, very bad graphic, not responsible, very bad game", 
                        "This app is so good, so fast, very responsible, so good application, very good game"]

            st.text('Bad Text: "i hate this game so much, this game is so bad, so slow, very bad graphic, not responsible, very bad game"')
            st.text('Good Text: "This app is so good, so fast, very responsible, so good application, very good game"')

            st.header("Hasil")
            for i in range(2):
                new_review = text_to_check[i]
                new_review = re.sub('[^a-zA-Z]', ' ', new_review)
                new_review = new_review.lower()
                new_review = new_review.split()
                lemmatizer = WordNetLemmatizer()
                all_stopwords = stopwords.words('english')
                new_review = [lemmatizer.lemmatize(word) for word in new_review if not word in set(all_stopwords)]
                new_review = ' '.join(new_review)
                new_corpus = [new_review]
                new_X_test = cv.transform(new_corpus).toarray()
                new_y_pred = classifier.predict(new_X_test)
                
                if new_y_pred == -1:
                    st.write(f"Teks \"{text_to_check[i]}\" merupakan teks yang negatif")
                else:
                    st.write(f"Teks \"{text_to_check[i]}\" merupakan teks yang positif")

            st.session_state["end_state"] = True

with st.form(key = 'form2'):
    if st.session_state["end_state"]:
        text_input_to_check = st.text_input(label ="Masukan beberapa kalimat....")
        predict_button = st.form_submit_button(label="Check")

        if predict_button:
            new_review = text_input_to_check
            new_review = re.sub('[^a-zA-Z]', ' ', new_review)
            new_review = new_review.lower()
            new_review = new_review.split()
            lemmatizer = WordNetLemmatizer()
            all_stopwords = stopwords.words('english')
            new_review = [lemmatizer.lemmatize(word) for word in new_review if not word in set(all_stopwords)]
            new_review = ' '.join(new_review)
            new_corpus = [new_review]
            new_X_test = st.session_state["cv_state"].transform(new_corpus).toarray()
            new_y_pred = st.session_state["classifier_state"].predict(new_X_test)
                
            if new_y_pred == -1:
                st.error("Teks tersebut merupakan teks yang negatif")
            else:
                st.success("Teks tersebut merupakan teks yang positif")