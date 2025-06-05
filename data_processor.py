import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def load_and_process_data(file_path):
    """医療データの読み込みと前処理を行う関数"""
    # CSVファイルを読み込む
    df = pd.read_csv(file_path)
    
    # 必要な特徴量を選択
    features = [
        '年齢', '性別', 'BMI', '血圧_最高', '血圧_最低',
        '運動頻度', '喫煙', '飲酒', '睡眠時間'
    ]
    
    targets = ['糖尿病', '高血圧', '心臓病']
    
    # 性別を数値に変換
    df['性別'] = df['性別'].map({'男性': 1, '女性': 0})
    
    # 特徴量とターゲットを分離
    X = df[features]
    y = df[targets]
    
    return X, y

def train_models(X, y):
    """複数の疾病に対するモデルの学習を行う関数"""
    # データを訓練用とテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 特徴量の標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 各疾病に対してモデルを学習
    models = {}
    for disease in y.columns:
        print(f"\n{disease}のモデル学習:")
        
        # モデルの学習
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train[disease])
        
        # モデルの評価
        y_pred = model.predict(X_test_scaled)
        print("\n分類レポート:")
        print(classification_report(y_test[disease], y_pred))
        
        # 特徴量の重要度
        feature_importance = pd.DataFrame({
            '特徴量': X.columns,
            '重要度': model.feature_importances_
        }).sort_values('重要度', ascending=False)
        print("\n特徴量の重要度:")
        print(feature_importance)
        
        models[disease] = model
    
    return models, scaler

def save_models(models, scaler, model_dir='models'):
    """モデルを保存する関数"""
    os.makedirs(model_dir, exist_ok=True)
    
    # 各疾病のモデルを保存
    for disease, model in models.items():
        model_path = os.path.join(model_dir, f'{disease}_model.joblib')
        joblib.dump(model, model_path)
    
    # スケーラーの保存
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)

def predict_risks(models, scaler, input_data):
    """健康リスクを予測する関数"""
    # 入力データの標準化
    input_scaled = scaler.transform(input_data)
    
    # 各疾病のリスクを予測
    risks = {}
    for disease, model in models.items():
        prob = model.predict_proba(input_scaled)[0][1]
        risks[disease] = prob
    
    return risks

def generate_sample_medical_data(n_samples=1000):
    """現実的な医療データのサンプルを生成する関数"""
    np.random.seed(42)
    
    data = {
        '年齢': np.random.randint(20, 80, n_samples),
        '性別': np.random.choice(['男性', '女性'], n_samples),
        '身長': np.random.normal(165, 10, n_samples),
        '体重': np.random.normal(60, 12, n_samples),
        '血圧_最高': np.random.normal(120, 15, n_samples),
        '血圧_最低': np.random.normal(80, 10, n_samples),
        '運動頻度': np.random.randint(0, 8, n_samples),
        '喫煙': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        '飲酒': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        '睡眠時間': np.random.normal(7, 1, n_samples)
    }
    
    # BMIの計算
    data['BMI'] = data['体重'] / ((data['身長']/100) ** 2)
    
    # 疾病リスクの生成（特徴量に基づいて）
    df = pd.DataFrame(data)
    
    # 糖尿病リスク
    diabetes_risk = (
        (df['BMI'] > 25) * 0.3 +
        (df['年齢'] > 50) * 0.2 +
        (df['運動頻度'] < 3) * 0.2 +
        df['喫煙'] * 0.1
    )
    data['糖尿病'] = (diabetes_risk + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    # 高血圧リスク
    hypertension_risk = (
        (df['血圧_最高'] > 130) * 0.3 +
        (df['血圧_最低'] > 85) * 0.3 +
        (df['BMI'] > 25) * 0.2 +
        df['喫煙'] * 0.1
    )
    data['高血圧'] = (hypertension_risk + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    # 心臓病リスク
    heart_disease_risk = (
        (df['血圧_最高'] > 140) * 0.3 +
        (df['年齢'] > 60) * 0.2 +
        (df['BMI'] > 30) * 0.2 +
        df['喫煙'] * 0.2
    )
    data['心臓病'] = (heart_disease_risk + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    # 医療データの生成（実際のデータに置き換えてください）
    print("医療データの生成中...")
    medical_data = generate_sample_medical_data(n_samples=10000)
    
    # データの保存
    os.makedirs('data/raw', exist_ok=True)
    medical_data.to_csv('data/raw/medical_data.csv', index=False)
    print("医療データを保存しました。")
    
    # データの読み込みと前処理
    print("\nデータの読み込みと前処理中...")
    X, y = load_and_process_data('data/raw/medical_data.csv')
    
    # モデルの学習
    print("\nモデルの学習中...")
    models, scaler = train_models(X, y)
    
    # モデルの保存
    print("\nモデルの保存中...")
    save_models(models, scaler)
    
    print("\n処理が完了しました。") 