## Dataset dan Topic
source: https://www.kaggle.com/datasets/jahnavipaliwal/field-of-study-vs-occupation/data

**Description**\
Dataset ini dirancang untuk membantu memprediksi apakah seseorang cenderung berganti pekerjaan berdasarkan latar belakangnya sebagai akademis, pengalaman kerja, dan faktor demografis lainnya. Dataset ini berisi 38,444 dan memiliki sekitar 22 atribut.

**Why is this dataset exciting?**\
Dataset ini dapat diterapkan untuk model machine learning agar dapat memprediksi perubahan karir berdasarkan berbagai jenis faktor. adapun dataset ini dapat digunakan untuk:
- Memprediksi mengapa orang ingin career change
- Membangun model prediktif untuk analisis SDM dan pengembangan karir
- Mempraktikkan machine learning models algorithms

**Some of the example questions that can be answered using this, for who?**
- Human Resource (HR): memprediksi potensi pergantian karyawan atau pola transisi karir
- Career Counseling: membantu individu untuk memahami perkembangan karir mereka dan membuat keputusan yang tepat nantinya di masa depan
- Analisis Industri: menganalisis tren pergeseran industri dan bidang pekerjaan yang sedang berkembang
- Job Market Insights: mendapatkan wawasan dan pemahaman mengenai latar belakang dari switch career/career change

```python
# function untuk membuat label pada bar chart
# menambahkan labels pada bar plots dan membuat axes lebih clean
# args:
# - axes     = matplotlib axes object
# - rotation = label rotation angle
# - location = label placement
def bar_labels(axes, rotation=0, location="edge"):
    for container in axes.containers:
        axes.bar_label(container, label_type=location, rotation=rotation)
    axes.set_yticklabels(())
    axes.set_xlabel("")
    axes.set_ylabel("")

# function utama untuk train model dan evaluasi performa
# args:
# - x_train: training feature data
# - y_train: training target data
# - x_test: testing feature data
# - y_test: testing target data
def training(x_train, y_train, x_test, y_test):
    # lists untuk store model performance metrics
    scores, reports, cms = [], dict(), dict()
    
    # diiterasi berdasarkan each model
    for model, name in zip(models, names):
        model.fit(x_train, y_train)    # train model
        pred = model.predict(x_test)   # membuat prediction
        
        # menghitung dan menyimpan performance metrics di dalam variable index
        scores.append(accuracy_score(pred, y_test) * 100)
        reports[name] = classification_report(pred, y_test)
        cms[name] = confusion_matrix(pred, y_test)
        
    # membuat DataFrame dari model scores
    results = pd.DataFrame({"score": scores}, index=names)
    results = results.sort_values("score", ascending=False)
    results["score"] = round(results["score"], 2)
    
    # visualize model accuracy scores
    fig, ax = plt.subplots()
    results["score"].plot(kind="bar", ax=ax)
    bar_labels(ax)
    plt.tight_layout()
    plt.show()
    
    # visualize confusion matrix
    index = 0
    # search proper ncols for dynamic used algorithm model
    half_length = len(names) // 2 + 1 if len(names) % 2 != 0 else len(names) // 2

    for _ in range(2):
        fig, axes = plt.subplots(ncols=half_length, figsize=(15, 6))
        for i in range(half_length):
            sns.heatmap(cms[results.index[index]], annot=True, fmt='d', ax=axes[i])
            axes[i].set_title(f"{results.index[index]}: {results.iloc[index, 0]}%")
            index += 1
        plt.tight_layout()
        plt.show()
    
    # tampilkan detailed classification report
    for name, report in reports.items():
        print("*" * 30)
        print(name)
        print(report)
        print("\n\n")

# load the dataset
df = pd.read_csv('career_change_prediction_dataset.csv')
# handle missing values in 'Family Influence' column
df["Family Influence"].fillna(df["Family Influence"].mode()[0], inplace=True)

# mengidentifikasi column categorical dan numerical
cats = [i for i in df.columns if df[i].nunique() <= 4]
cats = ["Field of Study", "Current Occupation"] + cats
nums = [i for i in df.columns if i not in cats]

# visualize distribution of categorical variables
index = 0
for _ in range(3):
    n = 4
    if _ == 2:
        n += 1
    fig, axes = plt.subplots(ncols=n, figsize=(15, 6))
    for i in range(n):
        df[cats[index]].value_counts().plot(kind="bar", ax=axes[i])
        bar_labels(axes[i])
        axes[i].set_title(cats[index])
        index += 1
    plt.tight_layout()
    plt.show()

# encode categorical variables
# :-1 dalam python artinya ambil semua elemen array kecuali elemen terakhir
for i in cats[:-1]:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i].values)

# scale numerical features
# melakukan scaling numeric menggunakan MinMaxScaler
# -1 artinya ambil kolom terakhir saja
scaler = MinMaxScaler()
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
x = scaler.fit_transform(x)

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

# run model training and evaluation
training(x_train, y_train, x_test, y_test)
```

## Model Algoritma yang Digunakan Untuk Dibandingkan
1. **Random Forest**
   
   **Alasan Penggunaan**:
   - Kemampuan Generalisasi: Kombinasi banyak pohon keputusan mengurangi risiko overfitting.
   - Feature Importance: Memberikan informasi fitur mana yang paling relevan.
   - Non-linear Relationships: Dapat menangkap pola kompleks dalam data.

   **Alasan Dibandingkan**:
   - Sebagai model ensemble yang kuat, membandingkannya dengan Gradient Boosting dan XGBoost memberikan wawasan tentang trade-off antara akurasi dan kecepatan pelatihan.

2. **Gradient Boosting**

   **Alasan Penggunaan**:
   - Efisiensi Boosting: Membangun model secara iteratif untuk memperbaiki kesalahan model sebelumnya.
   - Akurasi Tinggi: Cocok untuk data dengan pola yang kompleks.
   - Hyperparameter Tuning: Memberikan fleksibilitas untuk menyesuaikan model terhadap dataset.

   **Alasan Dibandingkan**:
   - Untuk melihat bagaimana Gradient Boosting dibandingkan dengan XGBoost dan LightGBM dalam hal kecepatan training dan accuracy.

3. **Logistic Regression**

   **Alasan Penggunaan**:
   - Baseline Model: Memberikan dasar performa untuk membandingkan model yang lebih kompleks.
   - Interpretasi Mudah: Hubungan linier antara fitur dan target mudah dipahami.
   - Efisiensi: Sangat cepat untuk dilatih dan cocok untuk dataset besar.
  
   **Alasan Dibandingkan**:
   - Untuk mengetahui sejauh mana model yang lebih kompleks (seperti Random Forest atau Gradient Boosting) meningkatkan performa dibanding model sederhana.
  
4. **Support Vector Machine**

   **Alasan Penggunaan**:
   - Non-linear Capabilities: Menggunakan kernel (misalnya, RBF) untuk menangkap relasi kompleks antara fitur dan target.
   - Margin Maximization: Dirancang untuk memaksimalkan margin antar kelas.
   - Efektivitas pada Dimensi Tinggi: Cocok jika dataset memiliki banyak atribut.

   **Alasan Dibandingkan**:
   - SVM sering lebih lambat dibandingkan ensemble methods (Random Forest, XGBoost). Membandingkannya dapat menunjukkan apakah kompleksitas tambahan SVM memberikan peningkatan signifikan.

5. **XGBoost**

   **Alasan Penggunaan**:
   - Optimasi Performa: Dikenal sebagai salah satu algoritma boosting tercepat dan paling efisien.
   - Regularization: Menyediakan regulasi bawaan untuk mengurangi overfitting.
   - Scalability: Dapat menangani dataset besar dengan efisiensi tinggi.

   **Alasan Dibandingkan**:
   - Untuk mengevaluasi keunggulan XGBoost dibandingkan Random Forest atau LightGBM dalam prediksi akurat dan kecepatan training.

6. **LightGBM**

   **Alasan Penggunaan**:
   - Kecepatan: Lebih cepat daripada XGBoost karena menggunakan teknik histogram-based splitting.
   - Scale: Cocok untuk dataset besar dan atribut dengan banyak kategori.
   - Feature Importance: Seperti Random Forest dan XGBoost, LightGBM juga memberikan wawasan fitur penting.

   **Alasan Dibandingkan**:
   - Untuk menilai performa LightGBM dibandingkan Gradient Boosting dan XGBoost dalam hal kecepatan dan accuracy.

- **Alasan Memilih Kombinasi Model tersebut**
  
   - Logistic Regression adalah baseline yang sederhana.
   - Random Forest adalah model ensemble yang mudah dipahami.
   - Gradient Boosting, XGBoost, dan LightGBM adalah metode boosting yang lebih kompleks.
   - SVM menawarkan pendekatan non-ensemble dengan kemampuan menangani pola non-linear.

Dengan membandingkan algoritma-algoritma ini, kita bisa menentukan algoritma mana yang memberikan performa optimal pada dataset ini, baik dari segi accuracy, kecepatan, maupun interpretabilitas.

## Analisis dan Evaluasi Model
1. **Algorithms with Perfect Scores**

   **[Random Forest, Gradient Boosting, XGBoost, LightGBM]**
   - Accuracy: 1.00
   - Precision, Recall, F1-Score: All are perfect (1.00) for both classes(0 & 1).
   - Suitability:
     - Hasil ini menunjukkan bahwa algoritma ini terlalu pas. Meskipun akurasi dan metrik lainnya sempurna, hal ini menimbulkan keraguuan tentang apakah model dapat men-generalisasi dengan baik untuk data yang tidak terlihat.
     - Overfitting sangat mungkin terjadi jika dataset tidak terlalu besar atau jika ada sedikit noise pada data.

2. **Logistic Regression**

   - Accuracy: 0.90
   - Precision, Recall:
     - Class 0: Precision and recall are 0.88.
     - Class 1: Precision and recall are ~0.91.
   - Suitability:
     - Logistic Regression adalah model yang lebih sederhana yang dapat men-generalisasi lebih baik daripada algoritma tree-based. Namun, akurasinya yang lebih rendah menunjukkan bahwa model ini kurang efektif ketika bekerja pada data yang kompleks.

3. **Support Vector Machine (SVM)**
   
   - Accuracy: 0.96
   - Precision, Recall:
     - Class 0: Precision and recall are ~0.95.
     - Class 1: Precision and recall are ~0.97.
   - Suitability:
     - SVM memiliki kinerja yang baik, tetapi tidak mencapai tingkat akurasi yang dicapai oleh model tree-based. Secara komputasi lebih mahal, terutama untuk set data yang besar, dan kurang dapat diinterpretasikan dibandingkan logistic regression atau model tree-based.

## Recommendation
Berdasarkan metrics yang digunakan, maka:
1. **Jika interpretabilitas dan generalisasi adalah prioritas**:
   
   Dapat menggunakan sebagai logistic regression sebagai model dasar. Akurasi yang lebih rendah diimbangi dengan kesederhanaan dan berkurangnya risiko overfitting.

2. **Jika memaksimalkan akurasi adalah goals-nya**

   Di antara model-model tree-based, XGBoost dan LightGBM cenderung men-generalisasi lebih baik daripada Random Forest atau Gradient Boosting karena mereka dioptimalkan untuk regularisasi dan menangani data terstruktur.

3. **Overfitting Concerns**

   Jika data pelatihan memiliki tingkat kesalahan yang rendah dan data uji memiliki tingkat kesalahan yang tinggi, itu menandakan adanya overfitting. Model seperti Random Forest dan Gradient Boosting harus dievaluasi lebih lanjut pada set tes yang tidak terlihat atau melalui cross-validation untuk mengkonfirmasi apakah score yang sempurna dapat digeneralisasi dengan baik.

4. **Feature Importance**

   Dapat menggunakan model seperti Random Forest, XGBoost, atau LightGBM untuk mengekstrak tingkat feature importance, yang dapat memberikan insights tentang attribute mana yang mendorong career change.

## Final Suggestion

Untuk balance antara performance dan interpretability, **XGBoost** atau **LightGBM** mungkin merupakan pilihan terbaik, asalkan dapat dipastikan bahwa keduanya tidak overfitting melalui langkah validasi tambahan.