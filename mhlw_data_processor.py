import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import numpy as np
import os

class MHLWDataProcessor:
    def load_sample_data(self):
        np.random.seed(42)
        n_samples = 1000

        data = {
            '年齢': np.random.normal(50, 15, n_samples),
            'BMI': np.random.normal(22, 3, n_samples),
            '性別': np.random.choice(['男性', '女性'], n_samples),
            '運動習慣': np.random.choice(['あり', 'なし'], n_samples, p=[0.3, 0.7]),
            '喫煙': np.random.choice(['吸う', '吸わない'], n_samples, p=[0.2, 0.8]),
        }

        self.data = pd.DataFrame(data)
        self.data['年齢'] = self.data['年齢'].astype(int)
        self.data = self.data[(self.data['年齢'] >= 20) & (self.data['年齢'] <= 90) & (self.data['BMI'] >= 15) & (self.data['BMI'] <= 40)]
        return self.data

    def compare_user_to_stats(self, bmi, age, gender):
        try:
            stats_df = pd.read_csv("bmi_stats.csv")
        except:
            return "統計データが読み込めませんでした。"

        def get_age_range(age):
            if 20 <= age < 30:
                return "20-29歳"
            elif 30 <= age < 40:
                return "30-39歳"
            elif 40 <= age < 50:
                return "40-49歳"
            elif 50 <= age < 60:
                return "50-59歳"
            elif 60 <= age < 70:
                return "60-69歳"
            else:
                return "70歳以上"

        def get_bmi_category(bmi, gender):
            if gender == "男性":
                return "BMI＜25、腹囲＜85ｃｍ" if bmi < 25 else "BMI≧25、腹囲≧85ｃｍ"
            else:
                return "BMI＜25、腹囲＜90ｃｍ" if bmi < 25 else "BMI≧25、腹囲≧90ｃｍ"

        age_range = get_age_range(age)
        category = get_bmi_category(bmi, gender)

        row = stats_df[
            (stats_df["性別"] == gender) &
            (stats_df["年齢層"] == age_range) &
            (stats_df["カテゴリ"].str.contains(category))
        ]

        if not row.empty:
            percentage = row["割合"].values[0]
            return f"{age_range}の{gender}のうち、{category}の人は {percentage:.1f}% です。"
        else:
            return "統計データに一致する項目が見つかりませんでした。"

    def generate_lifestyle_advice(self, bmi, age, gender):
        advice = {
            "運動": [],
            "食事": [],
            "生活習慣": [],
            "メンタルヘルス": []
        }

        if bmi < 18.5:
            advice["運動"] = [
                "医師と相談しながら軽めの運動を取り入れる",
                "筋力維持のためのストレッチや自重トレーニング",
                "疲労を感じたら無理せず休む"
            ]
            advice["食事"] = [
                "1日3食と間食を含めたエネルギー摂取を意識",
                "タンパク質を毎食取り入れる（卵・肉・魚・豆など）",
                "高カロリーかつ栄養価の高い食品を摂取（ナッツ・チーズなど）"
            ]
            advice["生活習慣"] = [
                "毎日同じ時間に食事・睡眠をとる",
                "体重・体調を定期的に記録する",
                "定期的に健康診断を受ける"
            ]
            advice["メンタルヘルス"] = [
                "体型への焦りを感じたら信頼できる人に相談",
                "SNS等の情報に過度に影響されない",
                "栄養士や心理カウンセラーへの相談も検討"
            ]

        elif 18.5 <= bmi < 25:
            advice["運動"] = [
                "週2〜3回のウォーキングやジョギングを継続",
                "柔軟体操やストレッチで姿勢改善も意識",
                "デスクワーク中心の場合は1時間に1回立ち上がる"
            ]
            advice["食事"] = [
                "主食・主菜・副菜を意識したバランスの良い食事",
                "水分を意識してこまめに摂る",
                "腹八分目を意識した食事量"
            ]
            advice["生活習慣"] = [
                "朝型生活を意識し、日中活動を活発にする",
                "適度なストレス解消法を見つける（趣味・運動など）",
                "睡眠の質を高める習慣（就寝前スマホ制限など）"
            ]
            advice["メンタルヘルス"] = [
                "過度に完璧を目指さず、自分を肯定する時間を作る",
                "周囲の支援や環境に感謝する習慣を意識"
            ]

        elif 25 <= bmi < 30:
            advice["運動"] = [
                "有酸素運動を週3回、1回30分程度から始める",
                "ストレッチやラジオ体操を毎朝の習慣にする",
                "階段を使う・一駅分歩くなど日常動作を増やす"
            ]
            advice["食事"] = [
                "野菜から食べ始めて血糖値の急上昇を抑える",
                "甘い飲料を控え、お茶や水に置き換える",
                "間食を1日1回以内に減らす"
            ]
            advice["生活習慣"] = [
                "毎朝・夜に体重を記録し可視化する",
                "夜更かしや深夜の食事を避ける",
                "1日8000歩以上を目標に歩数管理"
            ]
            advice["メンタルヘルス"] = [
                "目標体重を小さく区切って設定（例：-1kg/月）",
                "できたことを日記やアプリで記録して自信に変える"
            ]

        else:
            advice["運動"] = [
                "医師や専門家に相談の上、安全な運動計画を立てる",
                "プールでの水中運動や椅子体操など負荷の少ない活動を中心に",
                "一度にたくさん運動するより、短時間を毎日続けることを意識"
            ]
            advice["食事"] = [
                "栄養士の指導のもと1日の摂取カロリーを見直す",
                "加工食品や外食の頻度を減らす",
                "腹八分目で満足する訓練を意識する"
            ]
            advice["生活習慣"] = [
                "健康記録アプリを使って摂取量や活動を記録",
                "毎週決まった曜日に体重・体脂肪率をチェック",
                "就寝時間を一定に保ち、睡眠時間を6時間以上確保"
            ]
            advice["メンタルヘルス"] = [
                "短期的な結果に一喜一憂せず、長期視点で考える",
                "体型の変化だけでなく、気分・体調の変化にも注目する",
                "応援してくれる人を見つけ、感謝を共有する"
            ]

        return advice

    def load_csv_data(self, file_path):
        try:
            self.data = pd.read_csv(file_path, encoding='utf-8')
            return True
        except Exception as e:
            print(f"CSV読み込み失敗: {e}")
            return False

    def process_data(self):
        if self.data is None:
            print("データが読み込まれていません")
            return False
        self.data = self.data.dropna()
        return True

    def plot_bmi_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x='BMI', bins=30)
        plt.title('BMIの分布')
        plt.xlabel('BMI')
        plt.ylabel('頻度')
        plt.savefig('bmi_distribution.png')
        plt.close()

    def plot_bmi_by_gender(self):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.data, x='性別', y='BMI')
        plt.title('性別ごとのBMI分布')
        plt.savefig('bmi_by_gender.png')
        plt.close()

    def plot_age_bmi_relation(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x='年齢', y='BMI', hue='性別', alpha=0.5)
        plt.title('年齢とBMIの関係')
        plt.savefig('age_bmi_relation.png')
        plt.close()

    def generate_health_statistics(self):
        stats = {
            '全体': {
                'BMI平均': self.data['BMI'].mean(),
                'BMI中央値': self.data['BMI'].median(),
                'BMI標準偏差': self.data['BMI'].std(),
                '年齢平均': self.data['年齢'].mean(),
            }
        }

        for gender in self.data['性別'].unique():
            gender_data = self.data[self.data['性別'] == gender]
            stats[gender] = {
                'BMI平均': gender_data['BMI'].mean(),
                'BMI中央値': gender_data['BMI'].median(),
                'BMI標準偏差': gender_data['BMI'].std(),
                '年齢平均': gender_data['年齢'].mean(),
            }

        return stats
