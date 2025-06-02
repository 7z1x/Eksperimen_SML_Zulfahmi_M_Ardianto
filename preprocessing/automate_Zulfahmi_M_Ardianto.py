import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(test_size=0.2, random_state=42):
    print("Memulai preprocessing data...")
    print("Membaca file data.csv...")

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        data_path = os.path.join(base_dir, "..", "breast_cancer_dataset_raw", "data.csv")
        data_path = os.path.abspath(data_path) 

        df = pd.read_csv(data_path)
        print(f"File berhasil dibaca dari: {data_path}")
    except FileNotFoundError:
        print(f"Error: File data.csv tidak ditemukan di path yang diharapkan: {data_path}")
        print("Pastikan file 'data.csv' ada di dalam folder 'breast_cancer_dataset' (satu level di atas folder 'preprocessing').")
        return None, None, None, None, None
    except Exception as e:
        print(f"Terjadi error saat membaca data.csv: {e}")
        return None, None, None, None, None

    df_processed = df.copy()

    if 'diagnosis' in df_processed.columns and df_processed['diagnosis'].dtype == 'object':
        print("Melakukan label encoding pada kolom 'diagnosis'...")
        df_processed['diagnosis'] = df_processed['diagnosis'].map({'M': 1, 'B': 0})
    elif 'diagnosis' not in df_processed.columns:
        print("Error: Kolom 'diagnosis' tidak ditemukan dalam dataset.")
        return None, None, None, None, None

    print("Menghapus kolom 'id' dan 'Unnamed: 32'...")
    columns_to_drop = []
    if 'id' in df_processed.columns:
        columns_to_drop.append('id')
    if 'Unnamed: 32' in df_processed.columns:
        columns_to_drop.append('Unnamed: 32')

    if columns_to_drop:
        df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')

    print("Memisahkan fitur (X) dan target (y)...")
    try:
        X = df_processed.drop(columns=['diagnosis'])
        y = df_processed['diagnosis']
    except KeyError:
        print("Error: Gagal memisahkan fitur dan target. Pastikan kolom 'diagnosis' ada setelah preprocessing.")
        return None, None, None, None, None


    print("Melakukan split data train dan test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Melakukan standarisasi fitur...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    output_folder_name = "breast_cancer_dataset_preprocessed"
    preprocessed_dir = os.path.join(base_dir, output_folder_name)

    os.makedirs(preprocessed_dir, exist_ok=True)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    print(f"Menyimpan data yang sudah diproses ke folder: {preprocessed_dir}")
    X_train_scaled_df.to_csv(os.path.join(preprocessed_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled_df.to_csv(os.path.join(preprocessed_dir, "X_test_scaled.csv"), index=False)

    y_train_df = y_train.to_frame(name='diagnosis')
    y_test_df = y_test.to_frame(name='diagnosis')
    y_train_df.to_csv(os.path.join(preprocessed_dir, "y_train.csv"), index=False)
    y_test_df.to_csv(os.path.join(preprocessed_dir, "y_test.csv"), index=False)

    scaler_path = os.path.join(preprocessed_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler berhasil disimpan di: {scaler_path}")

    if isinstance(X, pd.DataFrame):
        feature_names_path = os.path.join(preprocessed_dir, "feature_names.txt")
        with open(feature_names_path, 'w') as f:
            for feature_name in X.columns:
                f.write(f"{feature_name}\n")
        print(f"Nama fitur berhasil disimpan di: {feature_names_path}")

    print("Data yang sudah diproses berhasil disimpan!")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    print("Program dimulai...")
    results = preprocess_data()
    if results is not None and results[0] is not None:
        print("Preprocessing selesai dengan sukses.")
    else:
        print("Preprocessing gagal atau tidak menghasilkan output.")
    print("Program selesai!")
