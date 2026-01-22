import streamlit as st
import sqlite3
import json
import os
import pandas as pd
from collections import Counter
import unicodedata
from khmernltk import word_tokenize
from datetime import datetime

# ========== PAGE CONFIG MUST BE FIRST ==========
st.set_page_config(
    page_title="Khmer Stop-Word Removal System",
    page_icon="📝",
    layout="wide"
)

# ========== CORE FUNCTIONS ==========
def normalize_text(text):
    normalized = unicodedata.normalize('NFKC', text)
    return normalized.strip()

class KhmerSegmenter:
    def __init__(self):
        pass
    
    def segment(self, text):
        try:
            tokens = word_tokenize(text)
            return [token.strip() for token in tokens if token.strip()]
        except Exception as e:
            st.error(f"Segmentation error: {e}")
            return []

class KhmerStopwordRemover:
    def __init__(self, stopwords_path="data/stopwords/final_stopword_list.txt"):
        self.segmenter = KhmerSegmenter()
        self.stopwords = self.load_stopwords(stopwords_path)
    
    def load_stopwords(self, filepath):
        stopwords = set()
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    stopwords.add(line.strip())
        return stopwords
    
    def remove_stopwords(self, text):
        normalized = normalize_text(text)
        tokens = self.segmenter.segment(normalized)
        filtered = [token for token in tokens if token not in self.stopwords]
        return filtered
    
    def Frequency(self, tokens, top_n=10):
        token_counts = Counter(tokens)
        # Get top N most frequent words
        top_words = dict(sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
        return dict(token_counts), top_words
    
    def linguistic_features(self, tokens):
        unique_tokens = set(tokens)
        duplicate_count = len(tokens) - len(unique_tokens)
        return {
            'total_tokens': len(tokens),
            'unique_tokens': len(unique_tokens),
            'duplicate_words': duplicate_count,
            'duplicate_percentage': round((duplicate_count / len(tokens) * 100), 2) if tokens else 0
        }

# ========== ENHANCED DATABASE FUNCTIONS ==========
def init_database():
    """Initialize SQLite database with enhanced schema"""
    conn = sqlite3.connect('khmer_nlp.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Enhanced table to store all analysis data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_text TEXT NOT NULL,
            original_tokens TEXT,
            filtered_tokens TEXT,
            removed_tokens TEXT,
            all_frequency_stats TEXT,
            top_frequency_stats TEXT,
            linguistic_stats TEXT,
            summary_stats TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            text_length INTEGER,
            original_word_count INTEGER,
            filtered_word_count INTEGER,
            removed_word_count INTEGER,
            reduction_percentage REAL
        )
    ''')
    
    # Create indexes for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON analysis_history(timestamp)')
    
    conn.commit()
    conn.close()

def save_analysis_to_db(text, result, original_tokens, all_freq, top_freq, linguistic_stats):
    """Save complete analysis to database"""
    conn = sqlite3.connect('khmer_nlp.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Calculate summary stats
    original_word_count = len(original_tokens)
    filtered_word_count = len(result['filtered_text'])
    removed_word_count = len(result['removed_text'])
    reduction_percentage = result['stats']['reduction_percentage']
    
    summary_stats = {
        'original_words': original_word_count,
        'filtered_words': filtered_word_count,
        'removed_words': removed_word_count,
        'reduction_percentage': reduction_percentage,
        'linguistic_features': {
            'total_tokens': linguistic_stats['total_tokens'],
            'unique_tokens': linguistic_stats['unique_tokens'],
            'duplicate_words': linguistic_stats['duplicate_words'],
            'duplicate_percentage': linguistic_stats['duplicate_percentage']
        },
        'top_words': list(top_freq.keys())
    }
    
    cursor.execute('''
        INSERT INTO analysis_history 
        (original_text, original_tokens, filtered_tokens, removed_tokens, 
         all_frequency_stats, top_frequency_stats, linguistic_stats, summary_stats,
         text_length, original_word_count, filtered_word_count, removed_word_count, reduction_percentage)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        text,
        json.dumps(original_tokens, ensure_ascii=False),
        json.dumps(result['filtered_text'], ensure_ascii=False),
        json.dumps(result['removed_text'], ensure_ascii=False),
        json.dumps(all_freq, ensure_ascii=False),
        json.dumps(top_freq, ensure_ascii=False),
        json.dumps(linguistic_stats, ensure_ascii=False),
        json.dumps(summary_stats, ensure_ascii=False),
        len(text),
        original_word_count,
        filtered_word_count,
        removed_word_count,
        reduction_percentage
    ))
    
    conn.commit()
    conn.close()
    return True

def get_all_history():
    """Retrieve all analysis history"""
    conn = sqlite3.connect('khmer_nlp.db', check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, original_text, timestamp, original_word_count, 
               filtered_word_count, removed_word_count, reduction_percentage,
               summary_stats
        FROM analysis_history 
        ORDER BY timestamp DESC
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row[0],
            'original_text': row[1],
            'timestamp': row[2],
            'original_word_count': row[3],
            'filtered_word_count': row[4],
            'removed_word_count': row[5],
            'reduction_percentage': row[6],
            'summary_stats': json.loads(row[7]) if row[7] else {}
        })
    
    return history

def get_analysis_by_id(analysis_id):
    """Get detailed analysis by ID"""
    conn = sqlite3.connect('khmer_nlp.db', check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM analysis_history WHERE id = ?', (analysis_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row[0],
            'original_text': row[1],
            'original_tokens': json.loads(row[2]) if row[2] else [],
            'filtered_tokens': json.loads(row[3]) if row[3] else [],
            'removed_tokens': json.loads(row[4]) if row[4] else [],
            'all_frequency_stats': json.loads(row[5]) if row[5] else {},
            'top_frequency_stats': json.loads(row[6]) if row[6] else {},
            'linguistic_stats': json.loads(row[7]) if row[7] else {},
            'summary_stats': json.loads(row[8]) if row[8] else {},
            'timestamp': row[9],
            'text_length': row[10],
            'original_word_count': row[11],
            'filtered_word_count': row[12],
            'removed_word_count': row[13],
            'reduction_percentage': row[14]
        }
    return None

# ========== PAGE FUNCTIONS ==========
def home_page(remover):
    """Display the main analysis page"""
    st.title("📝 Khmer Stop-Word Removal System")
    st.caption("Remove stopwords from Khmer text and analyze the results")
    
    # Initialize session state for current analysis
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'original_text' not in st.session_state:
        st.session_state.original_text = ""
    if 'original_tokens' not in st.session_state:
        st.session_state.original_tokens = []
    if 'clear_text' not in st.session_state:
        st.session_state.clear_text = False
    
    # Handle clear text functionality
    if st.session_state.clear_text:
        # Use a unique key for the text area when clearing
        text_area_key = "input_text_cleared"
        text = st.text_area(
            "**Enter Khmer Text:**",
            height=200,
            placeholder="សូមបញ្ចូលអត្ថបទភាសាខ្មែររបស់អ្នកនៅទីនេះ...",
            key=text_area_key,
            value=""
        )
        # Reset the clear flag
        st.session_state.clear_text = False
    else:
        # Normal text area
        text = st.text_area(
            "**Enter Khmer Text:**",
            height=200,
            placeholder="សូមបញ្ចូលអត្ថបទភាសាខ្មែររបស់អ្នកនៅទីនេះ...",
            key="input_text"
        )
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.write("**📈 Text Statistics:**")
        if text:
            char_count = len(text)
            word_count = len(text.split())
            st.metric("Characters", char_count)
            st.metric("Words", word_count)
        else:
            st.info("Enter text to see statistics")
        
        # Clear button - FIXED: Use callback approach
        if st.button("🗑️ Clear Text", use_container_width=True, key="clear_button"):
            st.session_state.clear_text = True
            st.session_state.current_analysis = None
            st.session_state.original_text = ""
            st.session_state.original_tokens = []
            st.rerun()
    
    # Analyze button
    analyze_col1, analyze_col2 = st.columns([3, 1])
    with analyze_col1:
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True, key="analyze_home"):
            if not text.strip():
                st.error("⚠️ Please enter some Khmer text first!")
            else:
                with st.spinner("Analyzing text..."):
                    # Process text
                    original_tokens = remover.segmenter.segment(normalize_text(text))
                    filtered_tokens = remover.remove_stopwords(text)
                    removed_tokens = [t for t in original_tokens if t not in filtered_tokens]
                    
                    # Get frequency data
                    all_freq, top_freq = remover.Frequency(filtered_tokens, top_n=10)
                    
                    # Get linguistic features
                    linguistic_stats = remover.linguistic_features(filtered_tokens)
                    
                    # Calculate stats
                    removed_count = len(removed_tokens)
                    reduction_percentage = (removed_count / len(original_tokens) * 100) if original_tokens else 0
                    
                    # Prepare result
                    result = {
                        'filtered_text': filtered_tokens,
                        'removed_text': removed_tokens,
                        'frequency_tokens': all_freq,
                        'top_frequency': top_freq,
                        'linguistic_': linguistic_stats,
                        'stats': {
                            'original_tokens': len(original_tokens),
                            'filtered_tokens': len(filtered_tokens),
                            'removed_tokens': removed_count,
                            'reduction_percentage': round(reduction_percentage, 2)
                        }
                    }
                    
                    # Save to database
                    try:
                        save_analysis_to_db(text, result, original_tokens, all_freq, top_freq, linguistic_stats)
                        st.session_state.current_analysis = result
                        st.session_state.original_text = text
                        st.session_state.original_tokens = original_tokens
                        st.success("✅ Analysis complete! Results saved to database.")
                    except Exception as e:
                        st.error(f"❌ Database error: {str(e)}")
                        st.session_state.current_analysis = result
                        st.session_state.original_text = text
                        st.session_state.original_tokens = original_tokens
    
    with analyze_col2:
        if st.button("📥 Export All Data", use_container_width=True, key="export_all_home"):
            export_all_data()
    
    # Display current analysis if exists
    if st.session_state.current_analysis:
        display_current_analysis()

def display_current_analysis():
    """Display the current analysis results"""
    result = st.session_state.current_analysis
    original_text = st.session_state.original_text
    
    st.divider()
    st.subheader("📊 Analysis Results")
    
    # Stats in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Original Words", 
            result['stats']['original_tokens'],
            delta=None
        )
    
    with col2:
        st.metric(
            "Filtered Words", 
            result['stats']['filtered_tokens'],
            delta=f"-{result['stats']['removed_tokens']}"
        )
    
    with col3:
        st.metric(
            "Removed Words", 
            result['stats']['removed_tokens'],
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Reduction", 
            f"{result['stats']['reduction_percentage']}%"
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Original Text", 
        "✅ Filtered Tokens", 
        "🗑️ Removed Tokens", 
        "📈 Word Frequency", 
        "🔬 Detailed Analysis"
    ])
    
    with tab1:
        st.write("**Original Text:**")
        st.text_area("", original_text, height=150, disabled=True, key="original_text_display")
        
        st.write("**Original Tokens:**")
        cols = st.columns(6)
        for idx, token in enumerate(st.session_state.original_tokens[:36]):
            with cols[idx % 6]:
                st.info(token)
        if len(st.session_state.original_tokens) > 36:
            st.caption(f"... and {len(st.session_state.original_tokens) - 36} more tokens")
    
    with tab2:
        st.write(f"**{len(result['filtered_text'])} Filtered Tokens:**")
        cols = st.columns(6)
        for idx, token in enumerate(result['filtered_text'][:36]):
            with cols[idx % 6]:
                st.success(token)
        if len(result['filtered_text']) > 36:
            st.caption(f"... and {len(result['filtered_text']) - 36} more tokens")
    
    with tab3:
        st.write(f"**{len(result['removed_text'])} Removed Stopwords:**")
        cols = st.columns(6)
        for idx, token in enumerate(result['removed_text'][:36]):
            with cols[idx % 6]:
                st.error(token)
        if len(result['removed_text']) > 36:
            st.caption(f"... and {len(result['removed_text']) - 36} more tokens")
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 10 Most Frequent Words:**")
            for word, freq in result['top_frequency'].items():
                st.write(f"• **{word}**: {freq} times")
        
        with col2:
            # Create bar chart data
            freq_df = pd.DataFrame(
                list(result['top_frequency'].items()),
                columns=['Word', 'Frequency']
            )
            st.bar_chart(freq_df.set_index('Word'))
    
    with tab5:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Linguistic Features:**")
            st.write(f"• **Total Tokens**: {result['linguistic_']['total_tokens']}")
            st.write(f"• **Unique Tokens**: {result['linguistic_']['unique_tokens']}")
            st.write(f"• **Duplicate Words**: {result['linguistic_']['duplicate_words']}")
            st.write(f"• **Duplicate Percentage**: {result['linguistic_']['duplicate_percentage']}%")
            
            # Calculate token type ratio
            if result['linguistic_']['total_tokens'] > 0:
                unique_ratio = (result['linguistic_']['unique_tokens'] / result['linguistic_']['total_tokens']) * 100
                st.write(f"• **Vocabulary Richness**: {unique_ratio:.1f}%")
        
        with col2:
            st.write("**Summary Statistics:**")
            st.write(f"• **Stopword Removal Rate**: {result['stats']['reduction_percentage']}%")
            st.write(f"• **Words Remaining**: {result['stats']['filtered_tokens']}")
            st.write(f"• **Compression Ratio**: {result['stats']['filtered_tokens'] / result['stats']['original_tokens']:.2%}")
            
            # Save current analysis button
            if st.button("💾 Save This Analysis", use_container_width=True, key="save_current"):
                st.success("Analysis is already saved to database!")
        
        # Export options
        st.divider()
        st.write("**Export Options:**")
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if st.button("📄 Export as JSON", use_container_width=True, key="export_json_home"):
                export_data = {
                    'original_text': st.session_state.original_text,
                    'original_tokens': st.session_state.original_tokens,
                    'filtered_tokens': result['filtered_text'],
                    'removed_tokens': result['removed_text'],
                    'frequency_analysis': result['frequency_tokens'],
                    'top_frequent_words': result['top_frequency'],
                    'linguistic_features': result['linguistic_'],
                    'summary_statistics': result['stats'],
                    'timestamp': datetime.now().isoformat()
                }
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, ensure_ascii=False, indent=2),
                    file_name=f"khmer_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_json_home"
                )

def history_page():
    """Display analysis history page"""
    st.title("📜 Analysis History")
    
    # Get all history
    history = get_all_history()
    
    if not history:
        st.info("📭 No analysis history found. Go to the Home page to analyze some text!")
        if st.button("🏠 Go to Home", key="go_home_from_empty"):
            st.session_state.page = 'home'
            st.rerun()
        return
    
    # Display history in a nice format
    st.write(f"**Total Analyses: {len(history)}**")
    
    for item in history:
        with st.expander(f"**Analysis #{item['id']}** - {item['timestamp'][:19]}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Original", item['original_word_count'])
            
            with col2:
                st.metric("Filtered", item['filtered_word_count'])
            
            with col3:
                st.metric("Removed", item['removed_word_count'])
            
            with col4:
                st.metric("Reduction", f"{item['reduction_percentage']:.1f}%")
            
            # Show text preview
            st.write("**Text Preview:**")
            st.text(item['original_text'][:200] + ("..." if len(item['original_text']) > 200 else ""))
            
            # Action buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                if st.button(f"📋 View Details", key=f"view_{item['id']}"):
                    # Load and display full details
                    analysis = get_analysis_by_id(item['id'])
                    if analysis:
                        display_analysis_details(analysis)
            
            with btn_col2:
                if st.button(f"📄 Export", key=f"export_{item['id']}"):
                    export_single_analysis(item['id'])
            
            with btn_col3:
                if st.button(f"🗑️ Delete", key=f"delete_{item['id']}"):
                    if st.checkbox(f"Confirm delete analysis #{item['id']}?", key=f"confirm_{item['id']}"):
                        delete_analysis(item['id'])
                        st.success(f"Analysis #{item['id']} deleted!")
                        st.rerun()

def display_analysis_details(analysis):
    """Display detailed analysis from database"""
    st.subheader(f"📋 Detailed Analysis - ID: {analysis['id']}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original Words", analysis['original_word_count'])
    col2.metric("Filtered Words", analysis['filtered_word_count'])
    col3.metric("Removed Words", analysis['removed_word_count'])
    col4.metric("Reduction", f"{analysis['reduction_percentage']:.1f}%")
    
    # Original text
    st.write("**Original Text:**")
    st.text_area("", analysis['original_text'], height=100, disabled=True, key=f"original_{analysis['id']}")
    
    # Linguistic features
    st.write("**Linguistic Features:**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"• Total Tokens: {analysis['linguistic_stats'].get('total_tokens', 0)}")
        st.write(f"• Unique Tokens: {analysis['linguistic_stats'].get('unique_tokens', 0)}")
    with col2:
        st.write(f"• Duplicate Words: {analysis['linguistic_stats'].get('duplicate_words', 0)}")
        st.write(f"• Duplicate Percentage: {analysis['linguistic_stats'].get('duplicate_percentage', 0)}%")
    
    # Top frequent words
    st.write("**Top 10 Frequent Words:**")
    if analysis['top_frequency_stats']:
        freq_df = pd.DataFrame(
            list(analysis['top_frequency_stats'].items()),
            columns=['Word', 'Frequency']
        )
        st.dataframe(freq_df, use_container_width=True)
    
    st.divider()

def export_single_analysis(analysis_id):
    """Export a single analysis to JSON"""
    analysis = get_analysis_by_id(analysis_id)
    if analysis:
        export_data = {
            'analysis_id': analysis['id'],
            'timestamp': analysis['timestamp'],
            'original_text': analysis['original_text'],
            'original_tokens': analysis['original_tokens'],
            'filtered_tokens': analysis['filtered_tokens'],
            'removed_tokens': analysis['removed_tokens'],
            'all_frequency_stats': analysis['all_frequency_stats'],
            'top_frequency_stats': analysis['top_frequency_stats'],
            'linguistic_stats': analysis['linguistic_stats'],
            'summary_stats': analysis['summary_stats'],
            'text_length': analysis['text_length'],
            'original_word_count': analysis['original_word_count'],
            'filtered_word_count': analysis['filtered_word_count'],
            'removed_word_count': analysis['removed_word_count'],
            'reduction_percentage': analysis['reduction_percentage']
        }
        
        st.download_button(
            label=f"Download Analysis #{analysis_id}",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name=f"khmer_analysis_{analysis_id}_{analysis['timestamp'][:10]}.json",
            mime="application/json",
            key=f"download_{analysis_id}"
        )

def export_all_data():
    """Export all analysis data"""
    history = get_all_history()
    if history:
        export_data = {
            'total_analyses': len(history),
            'export_timestamp': datetime.now().isoformat(),
            'analyses': history
        }
        
        st.download_button(
            label="Download All Data",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name=f"khmer_analyses_full_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            key="download_all_data"
        )

def delete_analysis(analysis_id):
    """Delete an analysis from database"""
    conn = sqlite3.connect('khmer_nlp.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analysis_history WHERE id = ?", (analysis_id,))
    conn.commit()
    conn.close()

# ========== MAIN APP ==========
def main():
    # Initialize database
    init_database()
    remover = KhmerStopwordRemover()
    
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    # Sidebar navigation
    with st.sidebar:
        st.title("⚙️ Navigation")
        
        # Stats
        conn = sqlite3.connect('khmer_nlp.db', check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM analysis_history")
        total_analyses = cursor.fetchone()[0]
        cursor.execute("SELECT SUM(original_word_count) FROM analysis_history")
        total_words = cursor.fetchone()[0] or 0
        conn.close()
        
        st.metric("📊 Total Analyses", total_analyses)
        st.metric("🔤 Total Words Processed", total_words)
        st.metric("📋 Stopwords Loaded", len(remover.stopwords))
        
        st.divider()
        
        # Navigation buttons with proper state updates
        if st.button("🏠 Home", use_container_width=True, key="nav_home"):
            st.session_state.page = 'home'
            st.rerun()
            
        if st.button("📜 View History", use_container_width=True, key="nav_history"):
            st.session_state.page = 'history'
            st.rerun()
    
    # Display the appropriate page based on session state
    if st.session_state.page == 'history':
        history_page()
    else:
        home_page(remover)

if __name__ == "__main__":
    main()