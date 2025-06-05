# app.py
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
from mhlw_data_processor import MHLWDataProcessor
import numpy as np
from datetime import datetime
import json
import os
import hashlib

# ページ設定を最初に実行（他のstコマンドより前に配置）
st.set_page_config(
    page_title="健康データ分析",
    layout="wide"
)
# 定数の定義
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
USER_HISTORY_DIR = os.path.join(BASE_DIR, "user_history")
DEFAULT_VALUES = {
    'age': 30,
    'height': 170.0,
    'weight': 60.0,
    'gender': "男性"
}

def get_default_value(key):
    """デフォルト値を取得する関数"""
    return DEFAULT_VALUES.get(key)

def reset_values():
    """入力フォームの値をデフォルト値にリセットする関数"""
    for key in DEFAULT_VALUES:
        if key in st.session_state:
            st.session_state[key] = DEFAULT_VALUES[key]
    st.session_state.calculated = False

def init_user_data():
    """ユーザーデータの初期化を行う関数"""
    try:
        # users.jsonファイルの作成
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        if not os.path.exists(USERS_FILE):
            with open(USERS_FILE, "w", encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
        
        # user_historyディレクトリの作成
        os.makedirs(USER_HISTORY_DIR, exist_ok=True)

    except Exception as e:
        print(f"Error in init_user_data: {str(e)}")
        st.error("データ初期化中にエラーが発生しました")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    """ユーザー登録を行う関数"""
    try:
        if not username or not password:
            return False, "ユーザー名とパスワードは必須です"
        
        if len(username) < 3:
            return False, "ユーザー名は3文字以上にしてください"
        
        if len(password) < 6:
            return False, "パスワードは6文字以上にしてください"
        
        # users.jsonファイルの存在確認
        if not os.path.exists(USERS_FILE):
            with open(USERS_FILE, "w", encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
        
        with open(USERS_FILE, "r", encoding='utf-8') as f:
            users = json.load(f)
        
        if username in users:
            return False, "このユーザー名は既に使用されています"
        
        users[username] = {
            "password": hash_password(password),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        with open(USERS_FILE, "w", encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        
        # ユーザーの履歴ファイルを作成
        user_history_file = os.path.join(USER_HISTORY_DIR, f"{username}.json")
        if not os.path.exists(user_history_file):
            with open(user_history_file, "w", encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False)
        
        return True, "登録が完了しました"
    except Exception as e:
        print(f"Error in register_user: {str(e)}")
        return False, "登録中にエラーが発生しました"

def authenticate_user(username, password):
    """ユーザー認証を行う関数"""
    try:
        if not username or not password:
            return False, "ユーザー名とパスワードを入力してください"
        
        if not os.path.exists(USERS_FILE):
            return False, "ユーザーデータが見つかりません"
        
        with open(USERS_FILE, "r", encoding='utf-8') as f:
            users = json.load(f)
        
        if username not in users:
            return False, "ユーザー名が見つかりません"
        
        if users[username]["password"] != hash_password(password):
            return False, "パスワードが正しくありません"
        
        return True, "ログインに成功しました"
    except Exception as e:
        print(f"Error in authenticate_user: {str(e)}")
        return False, "認証中にエラーが発生しました"

def save_user_history(username, result):
    """ユーザーの診断履歴を保存する関数"""
    try:
        history_file = os.path.join(USER_HISTORY_DIR, f"{username}.json")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding='utf-8') as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                history = []
        
        history.append(result)
        
        with open(history_file, "w", encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error in save_user_history: {str(e)}")
        st.error("履歴の保存中にエラーが発生しました")

def load_user_history(username):
    """ユーザーの診断履歴を読み込む関数"""
    try:
        history_file = os.path.join(USER_HISTORY_DIR, f"{username}.json")
        
        if not os.path.exists(history_file):
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            with open(history_file, "w", encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return []
            
        with open(history_file, "r", encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error in load_user_history: {str(e)}")
        return []

def calculate_bmi_status(bmi, age, gender):
    """BMIステータスを計算する関数"""
    # 年齢による判定基準の調整
    if age < 18:
        if bmi < 16:
            return "痩せすぎ", "🔵", "#e3f2fd", "体重増加が必要かもしれません。"
        elif bmi < 17:
            return "痩せ気味", "🔵", "#e3f2fd", "もう少し体重を増やすことを検討してください。"
        elif bmi < 25:
            return "普通体重", "🟢", "#e8f5e9", "健康的な体重です。"
        elif bmi < 30:
            return "やや体重過多", "🟡", "#fff3e0", "適度な運動を心がけましょう。"
        else:
            return "体重過多", "🔴", "#ffebee", "生活習慣の改善を検討してください。"
    elif age >= 65:
        if bmi < 18.5:
            return "低体重", "🔵", "#e3f2fd", "栄養バランスの改善を検討してください。"
        elif bmi < 25:
            return "普通体重", "🟢", "#e8f5e9", "健康的な体重を維持できています。"
        elif bmi < 27:
            return "やや高め", "🟡", "#fff3e0", "現状維持か、緩やかな改善を目指しましょう。"
        else:
            return "高体重", "🟠", "#fbe9e7", "徐々に改善を目指しましょう。"
    else:
        # 一般成人の判定基準（日本肥満学会の基準に基づく）
        if bmi < 16:
            return "痩せすぎ", "🔵", "#e3f2fd", "医療機関での相談をお勧めします。"
        elif bmi < 17:
            return "痩せ", "🔵", "#e3f2fd", "体重増加を検討してください。"
        elif bmi < 18.5:
            return "軽度痩せ", "🔵", "#e3f2fd", "もう少し体重を増やすことを検討してください。"
        elif bmi < 25:
            return "普通体重", "🟢", "#e8f5e9", "健康的な体重です。このまま維持しましょう。"
        elif bmi < 30:
            return "肥満（1度）", "🟡", "#fff3e0", "生活習慣の見直しを検討してください。"
        elif bmi < 35:
            return "肥満（2度）", "🟠", "#fbe9e7", "計画的な改善をお勧めします。"
        elif bmi < 40:
            return "肥満（3度）", "🔴", "#ffebee", "医療機関での相談をお勧めします。"
        else:
            return "肥満（4度）", "🔴", "#ffebee", "至急、医療機関での相談をお勧めします。"

def validate_measurements(height, weight, age):
    """身長・体重・年齢の妥当性をチェックする関数"""
    messages = []
    
    # 身長のチェック
    if height < 140:
        messages.append("身長が140cm未満です。入力値をご確認ください。")
    elif height > 220:
        messages.append("身長が220cmを超えています。入力値をご確認ください。")
    
    # 体重のチェック
    if weight < 30:
        messages.append("体重が30kg未満です。入力値をご確認ください。")
    elif weight > 200:
        messages.append("体重が200kgを超えています。入力値をご確認ください。")
    
    # BMIの極端な値をチェック
    bmi = weight / ((height/100) ** 2)
    if bmi < 12:  # 生存限界とされるBMI
        messages.append("BMIが極端に低い値となっています。入力値をご確認ください。")
    elif bmi > 60:  # 現実的な上限
        messages.append("BMIが極端に高い値となっています。入力値をご確認ください。")
    
    return messages

def calculate_health_risks(bmi, age, gender):
    """健康リスクを計算する関数"""
    risks = {
        "糖尿病": 0.0,
        "高血圧": 0.0,
        "心臓病": 0.0
    }
    
    # 基本リスク計算（BMIベース）
    if bmi < 18.5:  # 低体重
        base_risks = {"糖尿病": 0.15, "高血圧": 0.1, "心臓病": 0.1}
    elif bmi < 25:  # 普通体重
        base_risks = {"糖尿病": 0.1, "高血圧": 0.1, "心臓病": 0.1}
    elif bmi < 30:  # 肥満（1度）
        base_risks = {"糖尿病": 0.2, "高血圧": 0.25, "心臓病": 0.2}
    elif bmi < 35:  # 肥満（2度）
        base_risks = {"糖尿病": 0.35, "高血圧": 0.4, "心臓病": 0.3}
    else:  # 肥満（3度以上）
        base_risks = {"糖尿病": 0.5, "高血圧": 0.6, "心臓病": 0.4}

    # 年齢による調整
    age_factor = max(0, (age - 30) / 50)  # 30歳を基準として年齢による影響を計算
    
    # 性別による調整
    gender_factors = {
        "男性": {"糖尿病": 1.1, "高血圧": 1.2, "心臓病": 1.3},
        "女性": {"糖尿病": 1.0, "高血圧": 1.0, "心臓病": 1.0}
    }

    # 最終リスク計算
    for disease in risks:
        base_risk = base_risks[disease]
        gender_factor = gender_factors[gender][disease]
        
        # リスク計算式の改善
        risk = base_risk * (1 + age_factor) * gender_factor
        
        # リスクの上限設定
        risks[disease] = min(0.95, risk)

    return risks

def generate_lifestyle_advice(bmi, age, gender):
    """BMI、年齢、性別に基づいて生活アドバイスを生成する関数"""
    advice = {
        "運動": [],
        "食事": [],
        "生活習慣": [],
        "メンタルヘルス": []
    }
    
    # BMIの詳細な区分に基づくアドバイス
    if bmi < 16.0:  # 重度の低体重
        advice["運動"].extend([
            "過度な有酸素運動は控えめにする",
            "筋力トレーニングを中心に（週2-3回）",
            "ストレッチで柔軟性を維持",
            "疲労を感じたらすぐに休憩を取る"
        ])
        advice["食事"].extend([
            "1日6回程度の少量頻回食",
            "良質なタンパク質を毎食摂取（肉、魚、卵、大豆製品）",
            "健康的な脂質を積極的に摂取（ナッツ類、アボカド、オリーブオイル）",
            "消化の良い炭水化物を選ぶ（白米、パン、パスタなど）"
        ])
        advice["生活習慣"].extend([
            "毎日の体重記録",
            "十分な睡眠時間の確保（最低7-8時間）",
            "定期的な医師の診察を受ける",
            "過度な運動や活動を避ける"
        ])
        advice["メンタルヘルス"].extend([
            "無理なダイエットは避ける",
            "体重増加のストレスを抱え込まない",
            "必要に応じて専門家に相談"
        ])
    
    elif bmi < 18.5:  # 低体重
        advice["運動"].extend([
            "適度な筋力トレーニング（週2-3回）",
            "軽い有酸素運動（ウォーキング等）",
            "ヨガや軽いストレッチ"
        ])
        advice["食事"].extend([
            "1日3食＋間食2回の規則正しい食事",
            "タンパク質を意識的に摂取",
            "栄養バランスの良い食事を心がける",
            "カロリー計算アプリの活用"
        ])
        advice["生活習慣"].extend([
            "規則正しい生活リズム",
            "定期的な体重管理",
            "適度な休息を取る"
        ])
        advice["メンタルヘルス"].extend([
            "健康的な体重管理を意識する",
            "周囲のサポートを受け入れる"
        ])
    
    elif bmi < 25:  # 普通体重
        advice["運動"].extend([
            "定期的な有酸素運動（週3-4回）",
            "筋力トレーニング（週2-3回）",
            "ストレッチや柔軟体操",
            "好きなスポーツを楽しむ"
        ])
        advice["食事"].extend([
            "バランスの良い食事",
            "適切な食事量の維持",
            "野菜を十分に摂取",
            "水分を十分に摂取"
        ])
        advice["生活習慣"].extend([
            "規則正しい生活リズムの維持",
            "定期的な健康診断",
            "適度な運動習慣の継続"
        ])
        advice["メンタルヘルス"].extend([
            "ストレス解消法を見つける",
            "趣味や運動で気分転換"
        ])
    
    elif bmi < 30:  # 肥満（1度）
        advice["運動"].extend([
            "有酸素運動を中心に（週4-5回）",
            "筋力トレーニングの併用",
            "ウォーキングから始める",
            "徐々に運動強度を上げる"
        ])
        advice["食事"].extend([
            "食事量の適正化",
            "糖質の摂取を控えめに",
            "野菜を先に食べる",
            "間食を控える",
            "食事記録をつける"
        ])
        advice["生活習慣"].extend([
            "毎日の体重記録",
            "階段を使う",
            "こまめに体を動かす"
        ])
        advice["メンタルヘルス"].extend([
            "無理のない目標設定",
            "小さな成功を褒める",
            "継続的な取り組みを心がける"
        ])
    
    else:  # 肥満（2度以上）
        advice["運動"].extend([
            "医師に相談の上で運動を開始",
            "低強度の有酸素運動から始める",
            "水中運動の検討",
            "徐々に運動時間を延ばす"
        ])
        advice["食事"].extend([
            "栄養士への相談",
            "食事内容の記録",
            "食べる速度を遅くする",
            "野菜を多く摂取",
            "糖質・脂質の制限"
        ])
        advice["生活習慣"].extend([
            "定期的な医師の診察",
            "毎日の体重・体調記録",
            "生活リズムの改善"
        ])
        advice["メンタルヘルス"].extend([
            "専門家のサポートを受ける",
            "家族や友人のサポートを得る",
            "焦らず着実に改善を目指す"
        ])
    
    # 年齢による調整
    if age > 65:
        advice["運動"] = [adv.replace("強度", "負荷の軽い") for adv in advice["運動"]]
        advice["運動"].append("関節に優しい運動を選ぶ")
        advice["生活習慣"].append("転倒予防に注意する")
    elif age < 25:
        advice["運動"].append("成長期に合わせた適度な運動")
        advice["食事"].append("成長に必要な栄養素の摂取")
    
    # 性別による調整
    if gender == "女性":
        advice["食事"].append("鉄分・カルシウムを意識的に摂取")
        advice["生活習慣"].append("月経周期に合わせた体調管理")
    else:
        advice["食事"].append("適切なタンパク質摂取を心がける")
    
    return advice

# アプリケーションの初期化
init_user_data()

# スタイル設定を更新
style = """
<style>
    .title {
        text-align: center;
        padding: 2rem 0;
        color: #1E88E5;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .input-section {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    .result-section {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0.5rem 0;
        color: #1E88E5;
    }
    .metric-label {
        color: #666;
        font-size: 1rem;
        font-weight: 500;
    }
    .risk-section {
        background-color: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .risk-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        color: #333;
        text-align: center;
    }
    .risk-item {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .risk-item:hover {
        transform: translateY(-2px);
    }
    .history-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e6e6e6;
        transition: transform 0.2s;
    }
    .history-card:hover {
        transform: translateY(-2px);
    }
    .history-date {
        color: #1E88E5;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
        font-weight: 500;
    }
    .input-group {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .action-button {
        background-color: #1E88E5;
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        border: none;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .action-button:hover {
        background-color: #1976D2;
    }
    .reset-button {
        background-color: #f8f9fa;
        color: #666;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .reset-button:hover {
        background-color: #e9ecef;
    }
    .stat-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .chart-container {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .gender-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
</style>
"""

# メインアプリケーションの実行
def main():
    # セッションステートの初期化
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'calculated' not in st.session_state:
        st.session_state.calculated = False
    if 'gender' not in st.session_state:
        st.session_state.gender = DEFAULT_VALUES['gender']

    # アプリケーションのタイトル
    st.markdown('<h1 class="title">🏥 健康データ分析・BMI予測</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">あなたの健康状態を分析し、将来のリスクを予測します</p>', unsafe_allow_html=True)

    if not st.session_state.logged_in:
        # ログイン/登録画面のスタイリング
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["ログイン", "新規登録"])
        
        with tab1:
            with st.form("login_form"):
                st.markdown('<h3 style="color: #1E88E5; margin-bottom: 1.5rem;">アカウントにログイン</h3>', unsafe_allow_html=True)
                login_username = st.text_input("ユーザー名")
                login_password = st.text_input("パスワード", type="password")
                login_submitted = st.form_submit_button("ログイン", use_container_width=True)
                
                if login_submitted:
                    if login_username and login_password:
                        success, message = authenticate_user(login_username, login_password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.username = login_username
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.error("ユーザー名とパスワードを入力してください")
        
        with tab2:
            with st.form("register_form"):
                st.markdown('<h3 style="color: #1E88E5; margin-bottom: 1.5rem;">新規アカウント登録</h3>', unsafe_allow_html=True)
                new_username = st.text_input("ユーザー名")
                new_password = st.text_input("パスワード", type="password")
                confirm_password = st.text_input("パスワード（確認）", type="password")
                register_submitted = st.form_submit_button("登録", use_container_width=True)
                
                if register_submitted:
                    if new_username and new_password and confirm_password:
                        if new_password == confirm_password:
                            success, message = register_user(new_username, new_password)
                            if success:
                                st.success(message)
                                st.session_state.logged_in = True
                                st.session_state.username = new_username
                                st.rerun()
                            else:
                                st.error(message)
                        else:
                            st.error("パスワードが一致しません")
                    else:
                        st.error("すべての項目を入力してください")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # ログアウトボタンのスタイリング
        col1, col2 = st.columns([10, 2])
        with col2:
            if st.button("ログアウト", type="secondary"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.rerun()

        # メインのタブ
        tab1, tab2 = st.tabs(["📊 診断", "📋 履歴"])

        with tab1:
            # 入力フォームと結果表示のレイアウト
            input_col, result_col = st.columns([4, 6])

            with input_col:
                st.markdown('<div class="input-section">', unsafe_allow_html=True)
                st.markdown('<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 2rem;"><img src="https://cdn-icons-png.flaticon.com/512/3209/3209265.png" style="width: 32px; height: 32px;"/> <span style="font-size: 1.5rem; color: #333;">測定データ入力</span></div>', unsafe_allow_html=True)
                
                with st.container():
                    st.markdown('<div class="input-group">', unsafe_allow_html=True)
                    gender = st.radio(
                        "性別を選択",
                        options=["男性", "女性"],
                        horizontal=True,
                        key='gender'
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="input-group">', unsafe_allow_html=True)
                    input_method = st.radio(
                        "入力方法を選択",
                        ["スライダー", "直接入力"],
                        horizontal=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                    if input_method == "直接入力":
                        # 年齢入力（整数のまま）
                        age = st.number_input(
                            "年齢",
                            min_value=18,
                            max_value=100,
                            value=get_default_value('age'),
                            step=1,
                            key='age',
                            help="18歳から100歳までの値を入力してください"
                        )

                        # 身長入力（float型に統一）
                        height = st.number_input(
                            "身長 (cm)",
                            min_value=120.0,
                            max_value=220.0,
                            value=get_default_value('height'),
                            step=0.1,
                            format="%.1f",
                            key='height',
                            help="120cmから220cmまでの値を入力してください"
                        )

                        # 体重入力（float型に統一）
                        weight = st.number_input(
                            "体重 (kg)",
                            min_value=30.0,
                            max_value=200.0,
                            value=get_default_value('weight'),
                            step=0.1,
                            format="%.1f",
                            key='weight',
                            help="30kgから200kgまでの値を入力してください"
                        )
                    else:
                        # スライダーでの入力
                        age = st.slider(
                            "年齢",
                            min_value=18,
                            max_value=100,
                            value=get_default_value('age'),
                            step=1,
                            key='age',
                            help="スライダーを動かして年齢を選択してください"
                        )

                        height = st.slider(
                            "身長 (cm)",
                            min_value=120.0,
                            max_value=220.0,
                            value=get_default_value('height'),
                            step=0.5,
                            key='height',
                            help="スライダーを動かして身長を選択してください"
                        )

                        weight = st.slider(
                            "体重 (kg)",
                            min_value=30.0,
                            max_value=200.0,
                            value=get_default_value('weight'),
                            step=0.5,
                            key='weight',
                            help="スライダーを動かして体重を選択してください"
                        )
                
                col1, col2 = st.columns(2)
                with col1:
                    calculate_button = st.button("診断結果を計算", type="primary", use_container_width=True)
                with col2:
                    reset_button = st.button("入力をリセット", type="secondary", use_container_width=True, on_click=reset_values)
                
                st.markdown('</div>', unsafe_allow_html=True)

            # 計算ボタンが押されたらセッションステートを更新
            if calculate_button:
                st.session_state.calculated = True
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                
                # 診断結果を作成
                bmi = weight / ((height/100) ** 2)
                
                # 入力値の妥当性チェック
                validation_messages = validate_measurements(height, weight, age)
                if validation_messages:
                    for msg in validation_messages:
                        st.warning(msg)
                        # 1. モデルを読み込む
                        model = joblib.load("model.pkl")  # または糖尿病_model.joblibなど

                        # 2. 入力をAIに渡す
                        X_input = [[height, weight, age]]
                        risk_pred = model.predict(X_input)[0]

                        # 3. 結果を表示する
                        if risk_pred == 1:
                            st.warning("あなたは健康リスクがある可能性があります")
                        else:
                            st.success("現在のところ健康リスクは低いです")
                
                # BMI判定
                status, color, bg_color, advice = calculate_bmi_status(bmi, age, gender)
                
                # 診断結果を作成
                result = {
                    'datetime': current_time,
                    'gender': gender,
                    'age': age,
                    'height': height,
                    'weight': weight,
                    'bmi': bmi,
                    'status': status,
                    'color': color,
                    'bg_color': bg_color,
                    'advice': advice
                }

                # ユーザーの履歴に保存
                save_user_history(st.session_state.username, result)
                
                st.rerun()

            # 右側（計算結果と統計）
            with result_col:
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                
                if not st.session_state.calculated:
                    # 初期表示
                    st.info('👆 左側で必要な情報を入力し、「診断結果を計算」ボタンを押してください。')
                else:
                    st.header("🎯 診断結果")
                    
                    # BMI計算
                    bmi = weight / ((height/100) ** 2)
                    
                    # BMI判定
                    status, color, bg_color, advice = calculate_bmi_status(bmi, age, gender)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">BMI</div>
                            <div class="metric-value">{bmi:.1f}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="metric-card" style="background-color: {bg_color}">
                            <div class="metric-label">判定</div>
                            <div class="metric-value">{color} {status}</div>
                            <div style="font-size: 0.9rem; margin-top: 0.5rem; color: #666;">{advice}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        standard_weight = 22 * ((height/100) ** 2)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">標準体重</div>
                            <div class="metric-value">{standard_weight:.1f}<span style="font-size: 1rem">kg</span></div>
                        </div>
                        """, unsafe_allow_html=True)

                    # 健康リスク予測セクション
                    st.markdown('<div class="risk-section">', unsafe_allow_html=True)
                    st.markdown('<div class="risk-title">💊 健康リスク予測</div>', unsafe_allow_html=True)
                    
                    risks = calculate_health_risks(bmi, age, gender)
                    
                    # リスクの表示を3列に分ける
                    risk_col1, risk_col2, risk_col3 = st.columns(3)
                    
                    risk_colors = {
                        "低": "#4CAF50",
                        "中": "#FFA726",
                        "高": "#EF5350"
                    }

                    for i, (disease, risk) in enumerate(risks.items()):
                        with [risk_col1, risk_col2, risk_col3][i]:
                            risk_percentage = risk * 100
                            
                            # リスクレベルの判定
                            if risk < 0.3:
                                risk_level = "低"
                            elif risk < 0.6:
                                risk_level = "中"
                            else:
                                risk_level = "高"
                                
                            risk_color = risk_colors[risk_level]
                            
                            st.markdown(f"""
                            <div class="risk-item">
                                <div style="font-weight: bold; margin-bottom: 0.5rem">{disease}</div>
                                <div style="font-size: 0.9rem; color: {risk_color}; margin-bottom: 0.5rem">
                                    リスクレベル: {risk_level}
                                </div>
                            """, unsafe_allow_html=True)
                            st.progress(risk)
                            st.markdown(f"""
                                <div style="text-align: right; color: #666;">{risk_percentage:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # リスクに関する注意書き
                    st.markdown("""
                    <div style="font-size: 0.8rem; color: #666; margin-top: 1rem; padding: 1rem; background-color: #f8f9fa; border-radius: 5px;">
                        ※ このリスク予測は一般的な統計データに基づく参考値です。実際の健康状態は、生活習慣、遺伝的要因、
                        その他の健康状態など、様々な要因によって異なります。詳しい健康診断については、医療機関にご相談ください。
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    # アドバイスセクション
                    st.markdown('<div class="risk-section" style="margin-top: 2rem;">', unsafe_allow_html=True)
                    st.markdown('<div class="risk-title">💡 生活アドバイス</div>', unsafe_allow_html=True)
                    
                    advice = generate_lifestyle_advice(bmi, age, gender)
                    
                    # アドバイスの表示を4列に分ける
                    advice_cols = st.columns(4)
                    
                    for i, (category, items) in enumerate(advice.items()):
                        with advice_cols[i]:
                            st.markdown(f"""
                            <div class="risk-item">
                                <div style="font-weight: bold; margin-bottom: 0.5rem; color: #1E88E5;">
                                    {category}
                                </div>
                                <ul style="list-style-type: none; padding-left: 0;">
                                    {"".join(f'<li style="margin-bottom: 0.5rem; font-size: 0.9rem;">• {item}</li>' for item in items)}
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

                    # 統計データ分析（折りたたみ可能）
                    st.markdown("---")
                    with st.expander("📈 統計データ分析を表示", expanded=False):
                        # データプロセッサーのインスタンス化
                        processor = MHLWDataProcessor()
                        
                        # データソース選択
                        data_source = st.radio(
                            "データソースを選択",
                            ["サンプルデータ", "CSVファイルをアップロード"]
                        )

                        if data_source == "サンプルデータ":
                            processor.load_sample_data()
                        else:
                            uploaded_file = st.file_uploader("CSVファイルをアップロード", type=['csv'])
                            if uploaded_file is not None:
                                processor.load_csv_data(uploaded_file)

                        # データ分析の表示
                        if processor.data is not None:
                            # 基本統計情報
                            stats = processor.generate_health_statistics()
                            stat_col1, stat_col2, stat_col3 = st.columns(3)
                            with stat_col1:
                                st.metric("全体のBMI平均", f"{stats['全体']['BMI平均']:.2f}")
                            with stat_col2:
                                st.metric("全体の年齢平均", f"{stats['全体']['年齢平均']:.1f}")
                            with stat_col3:
                                st.metric("データ数", f"{len(processor.data):,}")

                            # BMI分布のグラフ
                            st.subheader("BMIの分布")
                            fig_bmi = px.histogram(
                                processor.data,
                                x="BMI",
                                nbins=30,
                                title="BMIの分布"
                            )
                            # 現在のBMIを示す垂直線を追加
                            fig_bmi.add_vline(
                                x=bmi,
                                line_dash="dash",
                                line_color="red",
                                annotation_text="あなたのBMI",
                                annotation_position="top"
                            )
                            st.plotly_chart(fig_bmi, use_container_width=True)

                            # 性別ごとのBMI分布
                            st.subheader("性別ごとのBMI分布")
                            fig_gender = px.box(
                                processor.data,
                                x="性別",
                                y="BMI",
                                title="性別ごとのBMI分布"
                            )
                            # 現在のBMIを示す水平線を追加
                            fig_gender.add_hline(
                                y=bmi,
                                line_dash="dash",
                                line_color="red",
                                annotation_text="あなたのBMI",
                                annotation_position="right"
                            )
                            st.plotly_chart(fig_gender, use_container_width=True)

                            # 年齢とBMIの関係
                            st.subheader("年齢とBMIの関係")
                            fig_age_bmi = px.scatter(
                                processor.data,
                                x="年齢",
                                y="BMI",
                                color="性別",
                                title="年齢とBMIの関係"
                            )
                            # 現在の位置をプロット
                            fig_age_bmi.add_trace(
                                go.Scatter(
                                    x=[age],
                                    y=[bmi],
                                    mode="markers",
                                    marker=dict(
                                        size=15,
                                        color="red",
                                        symbol="star"
                                    ),
                                    name="あなたの位置"
                                )
                            )
                            st.plotly_chart(fig_age_bmi, use_container_width=True)

                        else:
                            st.info("データを読み込んでください。")
                    
                    # 右側のコンテンツラッパーを閉じる
                    st.markdown('</div>', unsafe_allow_html=True)

        # 履歴タブ
        with tab2:
            # ユーザーの履歴を読み込む
            history = load_user_history(st.session_state.username)
            
            if not history:
                st.info("まだ診断履歴がありません。")
            else:
                # 履歴を新しい順に表示
                for result in reversed(history):
                    with st.container():
                        st.markdown(f"""
                        <div class="history-card">
                            <div class="history-date">{result['datetime']}</div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                                <div>性別: {result['gender']}</div>
                                <div>年齢: {result['age']}歳</div>
                                <div>身長: {result['height']:.1f}cm</div>
                                <div>体重: {result['weight']:.1f}kg</div>
                            </div>
                            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                                <div style="font-size: 1.2rem; margin-right: 1rem;">
                                    BMI: <strong>{result['bmi']:.1f}</strong>
                                </div>
                                <div style="font-size: 1.2rem;">
                                    判定: <strong>{result['color']} {result['status']}</strong>
                                </div>
                            </div>
                            <div style="color: #666; font-style: italic;">
                                {result.get('advice', '判定結果に基づいて生活習慣の改善を検討してください。')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()