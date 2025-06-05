# model_trainer.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# より現実的なデータを生成
np.random.seed(42)
n_samples = 1000  # サンプル数を増やす

# 年齢の分布（20-70歳、正規分布）
age = np.random.normal(40, 10, n_samples)
age = np.clip(age, 20, 70).astype(int)

# 身長の分布（性別による違いを考慮）
# 日本人の平均身長を参考に
male_ratio = 0.5
male_height = np.random.normal(170, 5.5, int(n_samples * male_ratio))  # 男性
female_height = np.random.normal(157, 5.0, n_samples - int(n_samples * male_ratio))  # 女性
height = np.concatenate([male_height, female_height])
height = np.clip(height, 140, 190)

# BMIの分布（正規分布をベース）
target_bmi = np.random.normal(22, 3, n_samples)
target_bmi = np.clip(target_bmi, 16, 35)  # 現実的なBMI範囲

# BMIから体重を計算
weight = target_bmi * (height/100) ** 2
weight = np.clip(weight, 35, 120)

# データフレーム作成
df = pd.DataFrame({
    "身長": height.round(1),
    "体重": weight.round(1),
    "年齢": age,
    "BMI": target_bmi.round(1)
})

# BMI 25以上を健康リスク有りとする
df["クラス"] = (df["BMI"] >= 25).astype(int)

# 特徴量とラベル
X = df[["身長", "体重", "年齢"]]
y = df["クラス"]

# データ分割と学習
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# モデルの評価
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"学習データのスコア: {train_score:.3f}")
print(f"テストデータのスコア: {test_score:.3f}")

# データの分布を確認
print("\nデータの基本統計量:")
print(df.describe().round(2))

# クラスの分布を確認
class_dist = df["クラス"].value_counts(normalize=True)
print("\n健康リスクの分布:")
print(f"リスク低（BMI 25未満）: {class_dist[0]:.1%}")
print(f"リスク高（BMI 25以上）: {class_dist[1]:.1%}")

# モデルを保存
joblib.dump(model, "model.pkl")
print("\n✅ モデルを保存しました: model.pkl")
