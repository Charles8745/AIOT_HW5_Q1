"""
AI / Human 文章偵測器 (AI Text Authorship Detector)
Streamlit Application - Enhanced Version

Features:
- Bilingual support (English/Chinese)
- Multiple detection models
- Perplexity analysis
- Feature explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features import FeatureExtractor
from src.models import TFIDFDetector, RandomForestDetector, EnsembleDetector, get_confusion_matrix, get_roc_curve
from src.transformer_detector import TransformerDetector, MockTransformerDetector, get_detector
from src.perplexity import PerplexityCalculator, BurstinessCalculator, get_advanced_features
from src.groq_client import get_groq_client
from src.utils import get_dataset_stats, preprocess_text

# Page configuration
st.set_page_config(
    page_title="AI/Human Text Detector | AI文章偵測器",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_theme_css(dark_mode: bool) -> str:
    """Generate CSS based on theme mode."""
    if dark_mode:
        return """
        <style>
            /* Dark Mode Styles */
            .stApp {
                background-color: #0e1117;
                color: #fafafa;
            }
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #4dabf7;
                text-align: center;
                padding: 1rem 0;
            }
            .sub-header {
                font-size: 1.2rem;
                color: #adb5bd;
                text-align: center;
                margin-bottom: 2rem;
            }
            .ai-result {
                color: #ff6b6b;
                font-weight: bold;
                font-size: 1.5rem;
            }
            .human-result {
                color: #51cf66;
                font-weight: bold;
                font-size: 1.5rem;
            }
            .feature-box {
                background-color: #1a1d24;
                border: 1px solid #2d3139;
                border-radius: 10px;
                padding: 1rem;
                margin: 0.5rem 0;
            }
            .lang-badge {
                display: inline-block;
                padding: 0.2rem 0.5rem;
                border-radius: 5px;
                font-size: 0.8rem;
                margin-left: 0.5rem;
            }
            .lang-en { background-color: #1c3a5e; color: #74c0fc; }
            .lang-zh { background-color: #4a3728; color: #ffc078; }
            .stMetric {
                background-color: #1a1d24;
                border-radius: 10px;
                padding: 1rem;
            }
            .stTabs [data-baseweb="tab-list"] {
                background-color: #1a1d24;
            }
            .stTabs [data-baseweb="tab"] {
                color: #adb5bd;
            }
            .stTabs [aria-selected="true"] {
                color: #4dabf7;
            }
            div[data-testid="stExpander"] {
                background-color: #1a1d24;
                border: 1px solid #2d3139;
            }
            .theme-indicator {
                background-color: #1a1d24;
                color: #fafafa;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                display: inline-block;
                margin-bottom: 1rem;
            }
        </style>
        """
    else:
        return """
        <style>
            /* Light Mode Styles */
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #1f77b4;
                text-align: center;
                padding: 1rem 0;
            }
            .sub-header {
                font-size: 1.2rem;
                color: #666;
                text-align: center;
                margin-bottom: 2rem;
            }
            .ai-result {
                color: #ff4444;
                font-weight: bold;
                font-size: 1.5rem;
            }
            .human-result {
                color: #44aa44;
                font-weight: bold;
                font-size: 1.5rem;
            }
            .feature-box {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 1rem;
                margin: 0.5rem 0;
            }
            .lang-badge {
                display: inline-block;
                padding: 0.2rem 0.5rem;
                border-radius: 5px;
                font-size: 0.8rem;
                margin-left: 0.5rem;
            }
            .lang-en { background-color: #e3f2fd; color: #1565c0; }
            .lang-zh { background-color: #fff3e0; color: #e65100; }
            .theme-indicator {
                background-color: #f8f9fa;
                color: #333;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                display: inline-block;
                margin-bottom: 1rem;
            }
        </style>
        """


def get_plotly_template(dark_mode: bool) -> str:
    """Get Plotly template based on theme mode."""
    return "plotly_dark" if dark_mode else "plotly_white"


def apply_dark_theme_to_fig(fig, dark_mode: bool):
    """Apply dark/light theme to a Plotly figure."""
    if dark_mode:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,29,36,0.8)',
            font=dict(color='#fafafa')
        )
    else:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,249,250,0.8)',
            font=dict(color='#333')
        )
    return fig


# ============ Cached Resources ============

@st.cache_resource
def load_models():
    """Load and cache all detection models."""
    feature_extractor = FeatureExtractor()
    tfidf_detector = TFIDFDetector()
    rf_detector = RandomForestDetector()
    ensemble = EnsembleDetector()
    perplexity_calc = PerplexityCalculator(use_neural=False)
    
    return {
        'feature_extractor': feature_extractor,
        'tfidf': tfidf_detector,
        'rf': rf_detector,
        'ensemble': ensemble,
        'perplexity': perplexity_calc
    }


@st.cache_data
def load_data():
    """Load and cache the sample dataset."""
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'samples.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        languages = df['language'].tolist() if 'language' in df.columns else ['en'] * len(texts)
        return df, texts, labels, languages
    else:
        st.error(f"Dataset not found at {data_path}")
        return None, [], [], []


@st.cache_resource
def train_models(_models, texts, labels):
    """Train the models with the dataset."""
    if not texts or not labels:
        return None
    
    feature_extractor = _models['feature_extractor']
    tfidf_detector = _models['tfidf']
    rf_detector = _models['rf']
    
    # Train TF-IDF model
    tfidf_metrics = tfidf_detector.train(texts, labels)
    
    # Extract features for RF model
    features = feature_extractor.extract_batch(texts)
    rf_metrics = rf_detector.train(features, labels, feature_extractor.get_feature_names())
    
    return {
        'tfidf_metrics': tfidf_metrics,
        'rf_metrics': rf_metrics
    }


@st.cache_resource
def load_transformer_detector(model_key: str, use_mock: bool):
    """Load transformer detector."""
    return get_detector(use_mock=use_mock, model_name=model_key)


# ============ UI Text (Bilingual) ============

UI_TEXT = {
    'en': {
        'title': '🔍 AI / Human Text Detector',
        'subtitle': 'Explainable AI Text Authorship Detection System',
        'tab_overview': '📊 Overview',
        'tab_analyzer': '🔍 Text Analyzer',
        'tab_comparison': '📈 Model Comparison',
        'tab_features': '🎯 Feature Explorer',
        'input_text': 'Enter text to analyze:',
        'analyze_btn': '🔍 Analyze Text',
        'result_ai': 'AI-generated',
        'result_human': 'Human-written',
        'detected_lang': 'Detected Language',
        'ai_prob': 'AI Probability',
        'confidence': 'Confidence',
        'ensemble_result': 'Ensemble Result',
        'perplexity': 'Perplexity',
        'burstiness': 'Burstiness',
        'feature_analysis': 'Feature Analysis',
        'model_probs': 'Model Probabilities',
        'total_samples': 'Total Samples',
        'human_samples': 'Human Samples',
        'ai_samples': 'AI Samples',
        'avg_words': 'Avg Words',
        'settings': '⚙️ Settings',
        'model_weights': 'Ensemble Weights',
        'transformer_model': 'Transformer Model',
        'use_mock': 'Use Mock Model (faster)',
        'use_real': 'Use Real Model (more accurate)',
        'generate_ai': '🤖 Generate AI Text',
        'theme': '🎨 Theme',
        'dark_mode': '🌙 Dark Mode',
        'light_mode': '☀️ Light Mode'
    },
    'zh': {
        'title': '🔍 AI / 人類 文章偵測器',
        'subtitle': '可解釋的 AI 文章作者偵測系統',
        'tab_overview': '📊 總覽',
        'tab_analyzer': '🔍 文本分析',
        'tab_comparison': '📈 模型比較',
        'tab_features': '🎯 特徵探索',
        'input_text': '請輸入要分析的文章：',
        'analyze_btn': '🔍 開始分析',
        'result_ai': 'AI 生成',
        'result_human': '人類撰寫',
        'detected_lang': '偵測語言',
        'ai_prob': 'AI 機率',
        'confidence': '信心度',
        'ensemble_result': '綜合結果',
        'perplexity': '困惑度',
        'burstiness': '突發性',
        'feature_analysis': '特徵分析',
        'model_probs': '各模型機率',
        'total_samples': '總樣本數',
        'human_samples': '人類樣本',
        'ai_samples': 'AI 樣本',
        'avg_words': '平均詞數',
        'settings': '⚙️ 設定',
        'model_weights': '模型權重',
        'transformer_model': 'Transformer 模型',
        'use_mock': '使用模擬模型（較快）',
        'use_real': '使用真實模型（較準確）',
        'generate_ai': '🤖 生成 AI 文字',
        'theme': '🎨 主題',
        'dark_mode': '🌙 深色模式',
        'light_mode': '☀️ 淺色模式'
    }
}


def get_text(key: str, lang: str = 'en') -> str:
    """Get UI text in specified language."""
    return UI_TEXT.get(lang, UI_TEXT['en']).get(key, key)


# ============ Main Application ============

def main():
    """Main application function."""
    
    # Initialize session state for dark mode
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    # Language and Theme selector in sidebar
    with st.sidebar:
        ui_lang = st.selectbox(
            "🌐 Language / 語言",
            options=['en', 'zh'],
            format_func=lambda x: 'English' if x == 'en' else '中文'
        )
        
        # Dark mode toggle
        t_temp = lambda key: get_text(key, ui_lang)
        st.subheader(t_temp('theme'))
        dark_mode = st.toggle(
            t_temp('dark_mode') if st.session_state.dark_mode else t_temp('light_mode'),
            value=st.session_state.dark_mode,
            key='dark_mode_toggle'
        )
        st.session_state.dark_mode = dark_mode
        
        st.divider()
    
    # Apply theme CSS
    st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)
    
    t = lambda key: get_text(key, ui_lang)
    
    # Header
    st.markdown(f'<p class="main-header">{t("title")}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{t("subtitle")}</p>', unsafe_allow_html=True)
    
    # Load models and data
    models = load_models()
    df, texts, labels, languages = load_data()
    
    # Train models if data is available
    training_metrics = None
    if texts and labels:
        training_metrics = train_models(models, texts, labels)
    
    # Sidebar settings
    with st.sidebar:
        st.header(t('settings'))
        
        # Model weights
        st.subheader(t('model_weights'))
        tfidf_weight = st.slider("TF-IDF", 0.0, 1.0, 0.33, 0.01)
        rf_weight = st.slider("Random Forest", 0.0, 1.0, 0.33, 0.01)
        transformer_weight = st.slider("Transformer", 0.0, 1.0, 0.34, 0.01)
        
        # Normalize weights
        total_weight = tfidf_weight + rf_weight + transformer_weight
        if total_weight > 0:
            models['ensemble'].set_weights({
                'tfidf': tfidf_weight / total_weight,
                'rf': rf_weight / total_weight,
                'transformer': transformer_weight / total_weight
            })
        
        st.divider()
        
        # Transformer model settings
        st.subheader(t('transformer_model'))
        use_mock = st.checkbox(t('use_mock'), value=True)
        
        if not use_mock:
            transformer_model = st.selectbox(
                "Model",
                options=list(TransformerDetector.AVAILABLE_MODELS.keys()),
                format_func=lambda x: x.replace('-', ' ').title()
            )
        else:
            transformer_model = 'mock'
        
        st.divider()
        
        # Groq API
        st.subheader("🤖 Groq API")
        use_groq = st.checkbox("Enable Groq API", value=False)
        if use_groq:
            groq_api_key = st.text_input("API Key", type="password")
            if groq_api_key:
                os.environ['GROQ_API_KEY'] = groq_api_key
    
    # Load transformer detector
    transformer_detector = load_transformer_detector(
        transformer_model if not use_mock else 'mock',
        use_mock
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        t('tab_overview'), 
        t('tab_analyzer'), 
        t('tab_comparison'),
        t('tab_features')
    ])
    
    # ============ Tab 1: Overview ============
    with tab1:
        st.header(t('tab_overview'))
        
        if df is not None and len(df) > 0:
            # Stats cards
            col1, col2, col3, col4 = st.columns(4)
            
            stats = get_dataset_stats(texts, labels)
            
            # Count by language
            en_count = sum(1 for lang in languages if lang == 'en')
            zh_count = sum(1 for lang in languages if lang == 'zh')
            
            with col1:
                st.metric(t('total_samples'), stats['total_samples'])
            with col2:
                st.metric(t('human_samples'), stats['human_count'])
            with col3:
                st.metric(t('ai_samples'), stats['ai_count'])
            with col4:
                st.metric("🌐 EN / ZH", f"{en_count} / {zh_count}")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Class distribution
                fig_pie = px.pie(
                    values=[stats['human_count'], stats['ai_count']],
                    names=[t('result_human'), t('result_ai')],
                    title='Class Distribution / 類別分佈',
                    color_discrete_sequence=['#44aa44', '#ff4444']
                )
                apply_dark_theme_to_fig(fig_pie, st.session_state.dark_mode)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Language distribution
                lang_counts = pd.Series(languages).value_counts()
                fig_lang = px.pie(
                    values=lang_counts.values,
                    names=['English' if x == 'en' else '中文' for x in lang_counts.index],
                    title='Language Distribution / 語言分佈',
                    color_discrete_sequence=['#1f77b4', '#ff7f0e']
                )
                apply_dark_theme_to_fig(fig_lang, st.session_state.dark_mode)
                st.plotly_chart(fig_lang, use_container_width=True)
            
            # Feature comparison
            st.subheader("Feature Comparison: Human vs AI / 特徵比較：人類 vs AI")
            
            feature_extractor = models['feature_extractor']
            all_features = []
            
            for text, label, lang in zip(texts, labels, languages):
                features = feature_extractor.extract_features(text, lang)
                features['label'] = 'Human' if label == 0 else 'AI'
                features['language'] = 'English' if lang == 'en' else '中文'
                all_features.append(features)
            
            features_df = pd.DataFrame(all_features)
            
            # Key features boxplot
            key_features = ['type_token_ratio', 'char_entropy', 'formal_word_ratio', 
                           'lexical_density', 'sentence_length_variance']
            
            fig_box = make_subplots(rows=1, cols=len(key_features), 
                                    subplot_titles=key_features)
            
            for i, feature in enumerate(key_features, 1):
                for label, color in [('Human', '#44aa44'), ('AI', '#ff4444')]:
                    data = features_df[features_df['label'] == label][feature]
                    fig_box.add_trace(
                        go.Box(y=data, name=label, marker_color=color, showlegend=(i==1)),
                        row=1, col=i
                    )
            
            fig_box.update_layout(height=400, title_text="Key Features by Class / 主要特徵分佈")
            apply_dark_theme_to_fig(fig_box, st.session_state.dark_mode)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("No dataset available. Please check data/samples.csv")
    
    # ============ Tab 2: Text Analyzer ============
    with tab2:
        st.header(t('tab_analyzer'))
        
        # Input method
        input_method = st.radio(
            "Input Method / 輸入方式",
            ["📝 Enter Text / 輸入文字", "🤖 Generate AI Text / 生成 AI 文字"],
            horizontal=True
        )
        
        if "📝" in input_method:
            input_text = st.text_area(
                t('input_text'),
                height=200,
                placeholder="Paste or type text here... / 在此貼上或輸入文字..."
            )
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                topic = st.text_input(
                    "Topic / 主題", 
                    "The impact of AI on society / AI 對社會的影響"
                )
            with col2:
                style = st.selectbox("Style / 風格", ["academic", "casual", "professional"])
            
            if st.button(t('generate_ai')):
                with st.spinner("Generating... / 生成中..."):
                    groq_client = get_groq_client(use_mock=not use_groq)
                    result = groq_client.generate_essay(topic, style)
                    if result['success']:
                        st.session_state['generated_text'] = result['text']
                    else:
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
            
            input_text = st.text_area(
                "Generated Text / 生成的文字",
                value=st.session_state.get('generated_text', ''),
                height=200
            )
        
        # Analyze button
        if st.button(t('analyze_btn'), type="primary"):
            if input_text and len(input_text.strip()) > 10:
                with st.spinner("Analyzing... / 分析中..."):
                    # Preprocess
                    clean_text = preprocess_text(input_text)
                    
                    # Detect language
                    detected_lang = models['feature_extractor'].detect_language(clean_text)
                    lang_display = "English 🇺🇸" if detected_lang == 'en' else "中文 🇹🇼"
                    
                    # Get predictions
                    predictions = {}
                    
                    # TF-IDF
                    if training_metrics:
                        tfidf_result = models['tfidf'].predict(clean_text)
                        predictions['tfidf'] = tfidf_result
                    
                    # Random Forest
                    if training_metrics:
                        features = models['feature_extractor'].extract_features_array(clean_text, detected_lang)
                        rf_result = models['rf'].predict(features)
                        predictions['rf'] = rf_result
                    
                    # Transformer
                    transformer_result = transformer_detector.predict(clean_text)
                    predictions['transformer'] = transformer_result
                    
                    # Ensemble
                    ensemble_result = models['ensemble'].combine_predictions(predictions)
                    
                    # Perplexity & Burstiness
                    advanced_features = get_advanced_features(clean_text)
                    
                    # Display Results
                    st.divider()
                    
                    # Main result
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        final_prob = ensemble_result['ensemble_probability']
                        final_label = t('result_ai') if final_prob >= 0.5 else t('result_human')
                        result_class = 'ai-result' if final_prob >= 0.5 else 'human-result'
                        
                        st.markdown(f"""
                        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px;'>
                            <h2 class='{result_class}'>{final_label}</h2>
                            <p style='font-size: 2rem; margin: 0;'>{final_prob:.1%}</p>
                            <p style='color: #666;'>{t('confidence')}: {ensemble_result['confidence']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric(t('detected_lang'), lang_display)
                        st.metric(t('perplexity'), f"{advanced_features['perplexity_score']:.1f}")
                    
                    with col3:
                        st.metric(t('burstiness'), f"{advanced_features['burstiness']:.3f}")
                        burstiness_note = "Human-like ✓" if advanced_features['burstiness'] > 0 else "AI-like"
                        st.caption(burstiness_note)
                    
                    st.divider()
                    
                    # Model comparison
                    st.subheader(t('model_probs'))
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        tfidf_prob = predictions.get('tfidf', {}).get('ai_probability', 0)
                        st.metric("TF-IDF", f"{tfidf_prob:.1%}")
                    
                    with col2:
                        rf_prob = predictions.get('rf', {}).get('ai_probability', 0)
                        st.metric("Random Forest", f"{rf_prob:.1%}")
                    
                    with col3:
                        trans_prob = predictions.get('transformer', {}).get('ai_probability', 0)
                        st.metric("Transformer", f"{trans_prob:.1%}")
                    
                    with col4:
                        st.metric(t('ensemble_result'), f"{final_prob:.1%}")
                    
                    # Probability bar chart
                    model_names = ['TF-IDF', 'Random Forest', 'Transformer', 'Ensemble']
                    probs = [
                        predictions.get('tfidf', {}).get('ai_probability', 0),
                        predictions.get('rf', {}).get('ai_probability', 0),
                        predictions.get('transformer', {}).get('ai_probability', 0),
                        final_prob
                    ]
                    
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=model_names,
                        y=probs,
                        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                        text=[f"{p:.1%}" for p in probs],
                        textposition='outside'
                    ))
                    fig_bar.add_hline(y=0.5, line_dash="dash", line_color="red",
                                     annotation_text="Threshold / 閾值")
                    fig_bar.update_layout(
                        title=t('model_probs'),
                        yaxis_title=t('ai_prob'),
                        yaxis_range=[0, 1.15]
                    )
                    apply_dark_theme_to_fig(fig_bar, st.session_state.dark_mode)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Feature analysis
                    st.subheader(t('feature_analysis'))
                    
                    feature_dict = models['feature_extractor'].extract_features(clean_text, detected_lang)
                    descriptions = models['feature_extractor'].get_feature_descriptions()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Text Statistics / 文本統計**")
                        stats_features = ['word_count', 'sentence_count', 'avg_word_length', 
                                         'avg_sentence_length', 'chinese_char_ratio']
                        for feat in stats_features:
                            if feat in feature_dict:
                                val = feature_dict[feat]
                                desc = descriptions.get(feat, feat)
                                st.write(f"• {desc}: **{val:.2f}**")
                    
                    with col2:
                        st.markdown("**Linguistic Features / 語言特徵**")
                        ling_features = ['type_token_ratio', 'char_entropy', 'formal_word_ratio',
                                        'lexical_density', 'function_word_ratio']
                        for feat in ling_features:
                            if feat in feature_dict:
                                val = feature_dict[feat]
                                desc = descriptions.get(feat, feat)
                                st.write(f"• {desc}: **{val:.3f}**")
                    
                    # Radar chart
                    radar_features = ['type_token_ratio', 'char_entropy', 'function_word_ratio',
                                     'lexical_density', 'formal_word_ratio']
                    radar_values = [feature_dict.get(f, 0) for f in radar_features]
                    
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=radar_values + [radar_values[0]],
                        theta=radar_features + [radar_features[0]],
                        fill='toself',
                        name='Input Text'
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        title="Feature Profile / 特徵雷達圖"
                    )
                    apply_dark_theme_to_fig(fig_radar, st.session_state.dark_mode)
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # JSON output
                    with st.expander("📋 Full Results (JSON)"):
                        st.json({
                            'language': detected_lang,
                            'predictions': predictions,
                            'ensemble': ensemble_result,
                            'advanced_features': advanced_features,
                            'text_features': feature_dict
                        })
            else:
                st.warning("Please enter at least 10 characters / 請輸入至少 10 個字元")
    
    # ============ Tab 3: Model Comparison ============
    with tab3:
        st.header(t('tab_comparison'))
        
        if training_metrics:
            # Metrics table
            st.subheader("Training Metrics / 訓練指標")
            
            metrics_data = {
                'Model': ['TF-IDF + LogReg', 'Random Forest'],
                'Accuracy': [
                    training_metrics['tfidf_metrics']['accuracy'],
                    training_metrics['rf_metrics']['accuracy']
                ],
                'Precision': [
                    training_metrics['tfidf_metrics']['precision'],
                    training_metrics['rf_metrics']['precision']
                ],
                'Recall': [
                    training_metrics['tfidf_metrics']['recall'],
                    training_metrics['rf_metrics']['recall']
                ],
                'F1 Score': [
                    training_metrics['tfidf_metrics']['f1'],
                    training_metrics['rf_metrics']['f1']
                ],
                'ROC-AUC': [
                    training_metrics['tfidf_metrics']['roc_auc'],
                    training_metrics['rf_metrics']['roc_auc']
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Format as percentages
            for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']:
                metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Confusion matrices
            st.subheader("Confusion Matrices / 混淆矩陣")
            
            tfidf_preds = [models['tfidf'].predict(text)['label'] for text in texts]
            features_all = models['feature_extractor'].extract_batch(texts)
            rf_preds = [models['rf'].predict(f)['label'] for f in features_all]
            
            col1, col2 = st.columns(2)
            
            with col1:
                cm_tfidf = get_confusion_matrix(labels, tfidf_preds)
                fig_cm1 = px.imshow(
                    cm_tfidf,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Human', 'AI'],
                    y=['Human', 'AI'],
                    color_continuous_scale='Blues',
                    text_auto=True
                )
                fig_cm1.update_layout(title="TF-IDF Confusion Matrix")
                apply_dark_theme_to_fig(fig_cm1, st.session_state.dark_mode)
                st.plotly_chart(fig_cm1, use_container_width=True)
            
            with col2:
                cm_rf = get_confusion_matrix(labels, rf_preds)
                fig_cm2 = px.imshow(
                    cm_rf,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Human', 'AI'],
                    y=['Human', 'AI'],
                    color_continuous_scale='Oranges',
                    text_auto=True
                )
                fig_cm2.update_layout(title="Random Forest Confusion Matrix")
                apply_dark_theme_to_fig(fig_cm2, st.session_state.dark_mode)
                st.plotly_chart(fig_cm2, use_container_width=True)
            
            # ROC Curves
            st.subheader("ROC Curves")
            
            tfidf_probs = [models['tfidf'].predict(text)['ai_probability'] for text in texts]
            rf_probs = [models['rf'].predict(f)['ai_probability'] for f in features_all]
            
            fig_roc = go.Figure()
            
            fpr1, tpr1, _ = get_roc_curve(labels, tfidf_probs)
            fig_roc.add_trace(go.Scatter(
                x=fpr1, y=tpr1,
                mode='lines',
                name=f"TF-IDF (AUC={training_metrics['tfidf_metrics']['roc_auc']:.3f})"
            ))
            
            fpr2, tpr2, _ = get_roc_curve(labels, rf_probs)
            fig_roc.add_trace(go.Scatter(
                x=fpr2, y=tpr2,
                mode='lines',
                name=f"Random Forest (AUC={training_metrics['rf_metrics']['roc_auc']:.3f})"
            ))
            
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Random Classifier'
            ))
            
            fig_roc.update_layout(
                title="ROC Curves",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )
            apply_dark_theme_to_fig(fig_roc, st.session_state.dark_mode)
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.warning("Models not trained. Please ensure training data is available.")
    
    # ============ Tab 4: Feature Explorer ============
    with tab4:
        st.header(t('tab_features'))
        
        if training_metrics and df is not None:
            # Feature importance
            st.subheader("Feature Importance (Random Forest) / 特徵重要性")
            
            importance = models['rf'].get_feature_importance()
            importance_df = pd.DataFrame(importance[:15], columns=['Feature', 'Importance'])
            
            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Features / 前 15 重要特徵',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
            apply_dark_theme_to_fig(fig_importance, st.session_state.dark_mode)
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # TF-IDF top features
            st.subheader("TF-IDF Feature Weights / TF-IDF 特徵權重")
            
            col1, col2 = st.columns(2)
            
            top_features = models['tfidf'].get_top_features(n=15)
            
            with col1:
                st.markdown("**🤖 AI Indicators / AI 指標**")
                ai_df = pd.DataFrame(top_features['ai_indicators'], columns=['Feature', 'Weight'])
                fig_ai = px.bar(ai_df, x='Weight', y='Feature', orientation='h',
                               color='Weight', color_continuous_scale='Reds')
                fig_ai.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                apply_dark_theme_to_fig(fig_ai, st.session_state.dark_mode)
                st.plotly_chart(fig_ai, use_container_width=True)
            
            with col2:
                st.markdown("**👤 Human Indicators / 人類指標**")
                human_df = pd.DataFrame(top_features['human_indicators'], columns=['Feature', 'Weight'])
                fig_human = px.bar(human_df, x='Weight', y='Feature', orientation='h',
                                  color='Weight', color_continuous_scale='Greens_r')
                fig_human.update_layout(yaxis={'categoryorder': 'total descending'}, height=400)
                apply_dark_theme_to_fig(fig_human, st.session_state.dark_mode)
                st.plotly_chart(fig_human, use_container_width=True)
            
            # Feature distribution explorer
            st.subheader("Feature Distribution Explorer / 特徵分佈探索器")
            
            feature_extractor = models['feature_extractor']
            all_features = []
            
            for text, label, lang in zip(texts, labels, languages):
                features = feature_extractor.extract_features(text, lang)
                features['label'] = 'Human' if label == 0 else 'AI'
                features['language'] = 'EN' if lang == 'en' else 'ZH'
                all_features.append(features)
            
            features_df = pd.DataFrame(all_features)
            
            selected_feature = st.selectbox(
                "Select Feature / 選擇特徵",
                feature_extractor.get_feature_names()
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dist = px.histogram(
                    features_df,
                    x=selected_feature,
                    color='label',
                    barmode='overlay',
                    opacity=0.7,
                    color_discrete_map={'Human': '#44aa44', 'AI': '#ff4444'},
                    title=f'{selected_feature} Distribution / 分佈'
                )
                apply_dark_theme_to_fig(fig_dist, st.session_state.dark_mode)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                fig_box = px.box(
                    features_df,
                    x='label',
                    y=selected_feature,
                    color='label',
                    color_discrete_map={'Human': '#44aa44', 'AI': '#ff4444'},
                    title=f'{selected_feature} by Class / 類別分佈'
                )
                apply_dark_theme_to_fig(fig_box, st.session_state.dark_mode)
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Feature description
            descriptions = feature_extractor.get_feature_descriptions()
            st.info(f"**{selected_feature}**: {descriptions.get(selected_feature, 'No description')}")
            
            # Language comparison
            st.subheader("Language Comparison / 語言比較")
            
            fig_lang_box = px.box(
                features_df,
                x='language',
                y=selected_feature,
                color='label',
                color_discrete_map={'Human': '#44aa44', 'AI': '#ff4444'},
                title=f'{selected_feature} by Language / 依語言分類'
            )
            apply_dark_theme_to_fig(fig_lang_box, st.session_state.dark_mode)
            st.plotly_chart(fig_lang_box, use_container_width=True)
        else:
            st.warning("Models not trained. Please ensure training data is available.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🔍 AI/Human Text Detector - Explainable AI Demo System</p>
        <p>Built with Streamlit | Supports English & 中文 | Multi-model Ensemble</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
