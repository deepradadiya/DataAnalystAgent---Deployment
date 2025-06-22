import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from together import Together
import PyPDF2
import docx
from PIL import Image
import pytesseract
import io
import base64
import json
import re
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class DataAnalystAgent:
    """
    An intelligent data analyst agent that can:
    1. Process multiple file types (.csv, .xlsx, .pdf, .docx, .txt, images)
    2. Perform data analysis and visualization
    3. Answer questions about the data
    4. Generate insights and recommendations
    """
    
    def __init__(self, together_api_key: str):
        """Initialize the agent with Together.ai API key"""
        self.client = Together(api_key=together_api_key)
        # Verify the correct model name with Together.ai documentation
        self.model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"  # Update if incorrect
        self.conversation_history = []
        self.current_data = None
        self.data_info = {}
        
    def load_document(self, file_path: str, file_type: str = None) -> Dict[str, Any]:
        """
        Load and process different types of documents
        
        Args:
            file_path: Path to the file
            file_type: Type of file (csv, xlsx, pdf, docx, txt, image)
            
        Returns:
            Dictionary containing processed data and metadata
        """
        try:
            if file_type is None:
                file_type = file_path.split('.')[-1].lower()
            
            result = {"success": False, "data": None, "error": None, "type": file_type}
            
            if file_type == 'csv':
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
                
                result.update({
                    "success": True,
                    "data": df,
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict(),
                    "summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
                })
                self.current_data = df
                print(f"✅ CSV loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
                
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
                result.update({
                    "success": True,
                    "data": df,
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict(),
                    "summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
                })
                self.current_data = df
                print(f"✅ Excel loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
                
            elif file_type == 'pdf':
                text_content = self._extract_pdf_text(file_path)
                result.update({
                    "success": True,
                    "data": text_content,
                    "length": len(text_content),
                    "word_count": len(text_content.split())
                })
                print(f"✅ PDF loaded successfully: {len(text_content)} characters")
                
            elif file_type == 'docx':
                text_content = self._extract_docx_text(file_path)
                result.update({
                    "success": True,
                    "data": text_content,
                    "length": len(text_content),
                    "word_count": len(text_content.split())
                })
                print(f"✅ DOCX loaded successfully: {len(text_content)} characters")
                
            elif file_type == 'txt':
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                text_content = None
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            text_content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                
                if text_content is None:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text_content = f.read()
                
                result.update({
                    "success": True,
                    "data": text_content,
                    "length": len(text_content),
                    "word_count": len(text_content.split())
                })
                print(f"✅ TXT loaded successfully: {len(text_content)} characters")
                
            elif file_type in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
                text_content = self._extract_image_text(file_path)
                result.update({
                    "success": True,
                    "data": text_content,
                    "length": len(text_content),
                    "extracted_from": "OCR"
                })
                print(f"✅ Image processed successfully: {len(text_content)} characters extracted")
                
            self.data_info = result
            print(f"📊 Data info stored: {self.data_info}")
            return result
            
        except Exception as e:
            result["error"] = str(e)
            return result
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def _extract_image_text(self, file_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            return f"OCR extraction failed: {str(e)}"
    
    def analyze_data(self, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Perform comprehensive data analysis
        
        Args:
            analysis_type: Type of analysis (comprehensive, statistical, exploratory)
            
        Returns:
            Analysis results
        """
        print(f"🔍 Starting analysis... Current data: {self.current_data is not None}")
        print(f"📋 Data info available: {self.data_info}")
        
        if self.current_data is None:
            return {"error": "No structured data loaded. Please upload a CSV or Excel file for analysis."}
        
        df = self.current_data
        analysis = {}
        
        try:
            print(f"📊 Analyzing dataframe with shape: {df.shape}")
            
            analysis['basic_info'] = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'memory_usage': int(df.memory_usage(deep=True).sum()),
                'missing_values': df.isnull().sum().to_dict()
            }
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(f"📈 Found {len(numeric_cols)} numeric columns: {list(numeric_cols)}")
            
            if len(numeric_cols) > 0:
                analysis['statistical_summary'] = df[numeric_cols].describe().to_dict()
                
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    analysis['correlation_matrix'] = corr_matrix.to_dict()
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            print(f"📝 Found {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
            
            if len(categorical_cols) > 0:
                analysis['categorical_summary'] = {}
                for col in categorical_cols:
                    try:
                        analysis['categorical_summary'][col] = {
                            'unique_values': int(df[col].nunique()),
                            'top_values': df[col].value_counts().head().to_dict()
                        }
                    except Exception as e:
                        print(f"⚠️ Error analyzing column {col}: {e}")
            
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            
            analysis['data_quality'] = {
                'completeness': float((1 - missing_cells / total_cells) * 100) if total_cells > 0 else 100,
                'duplicate_rows': int(df.duplicated().sum()),
                'unique_row_percentage': float((df.drop_duplicates().shape[0] / df.shape[0]) * 100) if df.shape[0] > 0 else 100
            }
            
            print("✅ Analysis completed successfully!")
            return analysis
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def create_visualizations(self, chart_types: List[str] = None) -> Dict[str, Any]:
        """
        Create various visualizations based on the data
        
        Args:
            chart_types: List of chart types to create
            
        Returns:
            Dictionary of visualization objects
        """
        if self.current_data is None:
            return {"error": "No data loaded"}
        
        df = self.current_data
        visualizations = {}
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            if len(numeric_cols) > 0:
                fig_dist = make_subplots(
                    rows=min(3, len(numeric_cols)), 
                    cols=min(2, max(1, len(numeric_cols)//3 + 1)),
                    subplot_titles=numeric_cols[:6]
                )
                
                for i, col in enumerate(numeric_cols[:6]):
                    row = i // 2 + 1
                    col_pos = i % 2 + 1
                    fig_dist.add_trace(
                        go.Histogram(x=df[col], name=col, showlegend=False),
                        row=row, col=col_pos
                    )
                
                fig_dist.update_layout(title="Distribution of Numeric Variables")
                visualizations['distributions'] = fig_dist
                
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    fig_corr = px.imshow(
                        corr_matrix,
                        title="Correlation Heatmap",
                        color_continuous_scale="RdBu_r",
                        aspect="auto"
                    )
                    visualizations['correlation_heatmap'] = fig_corr
                
                if len(numeric_cols) >= 2:
                    fig_box = go.Figure()
                    for col in numeric_cols[:5]:
                        fig_box.add_trace(go.Box(y=df[col], name=col))
                    fig_box.update_layout(title="Box Plots for Numeric Variables")
                    visualizations['box_plots'] = fig_box
            
            if len(categorical_cols) > 0:
                for col in categorical_cols[:3]:
                    value_counts = df[col].value_counts().head(10)
                    fig_bar = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Top Values in {col}"
                    )
                    visualizations[f'categorical_{col}'] = fig_bar
            
            if len(numeric_cols) >= 2:
                fig_scatter = px.scatter_matrix(
                    df[numeric_cols[:4]], 
                    title="Scatter Matrix"
                )
                visualizations['scatter_matrix'] = fig_scatter
            
            return visualizations
            
        except Exception as e:
            return {"error": f"Visualization creation failed: {str(e)}"}
    
    def query_llm(self, prompt: str, context: str = None) -> str:
        """
        Query the Llama model with context about the data
        
        Args:
            prompt: User's question or request
            context: Additional context about the data
            
        Returns:
            Model's response
        """
        try:
            data_context = ""
            if self.current_data is not None:
                data_context = f"""
                Current dataset information:
                - Shape: {self.current_data.shape}
                - Columns: {', '.join(self.current_data.columns.tolist())}
                - Data types: {dict(self.current_data.dtypes)}
                - Sample data:\n{self.current_data.head().to_string()}
                
                Statistical summary:
                {self.current_data.describe().to_string() if len(self.current_data.select_dtypes(include=[np.number]).columns) > 0 else 'No numeric columns available'}
                """
            elif self.data_info.get('data') and isinstance(self.data_info['data'], str):
                data_context = f"""
                Current document content (first 1000 characters):
                {self.data_info['data'][:1000]}...
                
                Document info:
                - Type: {self.data_info.get('type', 'unknown')}
                - Length: {self.data_info.get('length', 0)} characters
                - Word count: {self.data_info.get('word_count', 0)}
                """
            
            full_prompt = f"""
            You are an expert data analyst. Answer the user's question based on the provided data context.
            Be specific, accurate, and provide actionable insights when possible.
            
            {data_context}
            
            Additional context: {context or 'None'}
            
            User question: {prompt}
            
            Please provide a comprehensive answer with specific examples from the data when relevant.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=2048,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            self.conversation_history.append({
                "question": prompt,
                "answer": answer,
                "timestamp": pd.Timestamp.now()
            })
            
            return answer
            
        except Exception as e:
            return f"Error querying model: {str(e)}"
    
    def generate_insights(self) -> str:
        """Generate automatic insights about the loaded data"""
        if self.current_data is None and not self.data_info.get('data'):
            return "No data loaded to generate insights."
        
        if self.current_data is not None:
            prompt = """
            Analyze the provided dataset and generate key insights including:
            1. Data quality assessment
            2. Key patterns and trends
            3. Anomalies or outliers
            4. Recommendations for further analysis
            5. Business implications (if applicable)
            
            Provide actionable insights that would be valuable for decision making.
            """
        else:
            prompt = f"""
            Analyze the provided document and generate key insights including:
            1. Main themes and topics
            2. Important findings or conclusions
            3. Data points or statistics mentioned
            4. Recommendations or action items
            5. Summary of key information
            
            Document type: {self.data_info.get('type', 'unknown')}
            """
        
        return self.query_llm(prompt)
    
    def export_analysis_report(self) -> str:
        """Export a comprehensive analysis report"""
        if self.current_data is None and not self.data_info.get('data'):
            return "No data available for report generation."
        
        report = "# Data Analysis Report\n\n"
        
        if self.current_data is not None:
            df = self.current_data
            report += f"## Dataset Overview\n"
            report += f"- **Rows**: {df.shape[0]:,}\n"
            report += f"- **Columns**: {df.shape[1]}\n"
            report += f"- **Column Names**: {', '.join(df.columns.tolist())}\n\n"
            
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            report += f"## Data Quality\n"
            report += f"- **Completeness**: {100-missing_pct:.1f}%\n"
            report += f"- **Duplicate Rows**: {df.duplicated().sum()}\n\n"
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                report += f"## Statistical Summary\n"
                report += df[numeric_cols].describe().to_string()
                report += "\n\n"
        
        else:
            report += f"## Document Overview\n"
            report += f"- **Type**: {self.data_info.get('type', 'unknown')}\n"
            report += f"- **Size**: {self.data_info.get('length', 0):,} characters\n"
            report += f"- **Words**: {self.data_info.get('word_count', 0):,}\n\n"
        
        report += "## AI-Generated Insights\n"
        report += self.generate_insights()
        report += "\n\n"
        
        if self.conversation_history:
            report += "## Q&A History\n"
            for i, qa in enumerate(self.conversation_history[-5:], 1):
                report += f"### Question {i}\n**Q**: {qa['question']}\n**A**: {qa['answer']}\n\n"
        
        return report

# Streamlit Application
def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Data Analyst Agent",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🤖 Intelligent Data Analyst Agent")
    st.markdown("*Powered by Together.ai*")
    
    # Debug: Display environment info
    st.write(f"Debug: Working directory: {os.getcwd()}")
    st.write(f"Debug: Streamlit config dir: {os.getenv('STREAMLIT_CONFIG_DIR', '~/.streamlit')}")
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Get API key from environment variable
    api_key = os.getenv("TOGETHER_API_KEY")
    if api_key:
        if st.session_state.agent is None:
            st.session_state.agent = DataAnalystAgent(api_key)
            st.success("Agent initialized!")
    else:
        st.error("⚠️ TOGETHER_API_KEY environment variable not set. Please contact the app administrator.")
        return
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png'],
            help="Supported formats: CSV, Excel, PDF, Word, Text, Images"
        )
        
        # File size limit (50MB)
        if uploaded_file and uploaded_file.size > 50 * 1024 * 1024:
            st.error("File size exceeds 50MB. Please upload a smaller file.")
            return
        
        if uploaded_file and st.session_state.agent:
            # Save uploaded file temporarily in /tmp
            temp_path = os.path.join("/tmp", f"temp_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Load document
            with st.spinner("Processing document..."):
                result = st.session_state.agent.load_document(temp_path)
                st.write("Debug - Load result:", result)
                
            if result['success']:
                st.success(f"✅ {uploaded_file.name} loaded successfully!")
                st.session_state.data_loaded = True
                
                if result.get('shape'):
                    st.info(f"📊 Shape: {result['shape'][0]} rows × {result['shape'][1]} columns")
                    st.write("Columns:", result.get('columns', []))
                elif result.get('word_count'):
                    st.info(f"📄 {result['word_count']} words, {result['length']} characters")
                
                if st.session_state.agent.current_data is not None:
                    st.success("✅ Structured data loaded and ready for analysis!")
                else:
                    st.info("📄 Document loaded (text-based, limited analysis available)")
                    
            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            # Clean up temp file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
    
    # Main content area
    if not st.session_state.data_loaded:
        st.info("👆 Please upload a document using the sidebar to begin analysis.")
        return
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🔍 Analysis", 
        "📈 Visualizations", 
        "💬 Q&A Chat", 
        "📋 Report"
    ])
    
    with tab1:
        st.header("Data Overview")
        
        if st.session_state.agent.current_data is not None:
            df = st.session_state.agent.current_data
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Duplicates", df.duplicated().sum())
            
            st.subheader("First 10 Rows")
            st.dataframe(df.head(10))
            
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info)
            
        else:
            st.subheader("Document Content Preview")
            content = st.session_state.agent.data_info.get('data', '')
            st.text_area("Content", content[:2000] + "..." if len(content) > 2000 else content, height=300)
    
    with tab2:
        st.header("Data Analysis")
        
        if st.checkbox("Show Debug Info"):
            st.write("Current data status:", st.session_state.agent.current_data is not None if st.session_state.agent else "No agent")
            if st.session_state.agent and st.session_state.agent.current_data is not None:
                st.write("Data shape:", st.session_state.agent.current_data.shape)
                st.write("Data columns:", st.session_state.agent.current_data.columns.tolist())
        
        if st.session_state.agent.current_data is None:
            st.warning("⚠️ Analysis requires structured data (CSV/Excel). Please upload a CSV or Excel file.")
            
            if st.session_state.agent.data_info.get('data'):
                st.info("📄 Text analysis available - try the Q&A tab for document insights!")
        else:
            if st.button("🔄 Run Comprehensive Analysis"):
                with st.spinner("Analyzing data..."):
                    st.session_state.analysis_results = st.session_state.agent.analyze_data()
                    st.write("Debug - Analysis results:", st.session_state.analysis_results)
        
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            if 'error' not in results:
                if 'statistical_summary' in results and results['statistical_summary']:
                    st.subheader("📊 Statistical Summary")
                    try:
                        st.dataframe(pd.DataFrame(results['statistical_summary']))
                    except Exception as e:
                        st.write("Statistical data:", results['statistical_summary'])
                
                if 'data_quality' in results:
                    st.subheader("✅ Data Quality Assessment")
                    quality = results['data_quality']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Completeness", f"{quality['completeness']:.1f}%")
                    with col2:
                        st.metric("Duplicate Rows", quality['duplicate_rows'])
                    with col3:
                        st.metric("Unique Rows", f"{quality['unique_row_percentage']:.1f}%")
                
                if 'basic_info' in results:
                    st.subheader("ℹ️ Basic Information")
                    info = results['basic_info']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Rows", f"{info['shape'][0]:,}")
                        st.metric("Total Columns", info['shape'][1])
                    with col2:
                        st.metric("Memory Usage", f"{info['memory_usage']/1024/1024:.1f} MB")
                        st.metric("Missing Values", sum(info['missing_values'].values()))
                
                if 'categorical_summary' in results and results['categorical_summary']:
                    st.subheader("📋 Categorical Variables")
                    for col, info in results['categorical_summary'].items():
                        with st.expander(f"Column: {col}"):
                            st.write(f"**Unique values**: {info['unique_values']}")
                            st.write("**Top values**:")
                            for value, count in info['top_values'].items():
                                st.write(f"  • {value}: {count}")
            else:
                st.error(f"Analysis failed: {results['error']}")
    
    with tab3:
        st.header("Data Visualizations")
        
        if st.session_state.agent.current_data is not None:
            if st.button("📈 Generate Visualizations"):
                with st.spinner("Creating visualizations..."):
                    viz_results = st.session_state.agent.create_visualizations()
                
                if 'error' not in viz_results:
                    for viz_name, fig in viz_results.items():
                        st.subheader(viz_name.replace('_', ' ').title())
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Visualization failed: {viz_results['error']}")
        else:
            st.info("Visualizations are available only for structured data (CSV/Excel files).")
    
    with tab4:
        st.header("💬 Interactive Q&A")
        
        user_question = st.text_input("Ask anything about your data:", placeholder="e.g., What are the main trends in this data?")
        
        if st.button("🚀 Ask") and user_question:
            with st.spinner("Thinking..."):
                answer = st.session_state.agent.query_llm(user_question)
            
            st.write("**Answer:**")
            st.write(answer)
        
        if st.session_state.agent.conversation_history:
            st.subheader("📜 Conversation History")
            for i, qa in enumerate(reversed(st.session_state.agent.conversation_history[-5:]), 1):
                with st.expander(f"Q{i}: {qa['question'][:50]}..."):
                    st.write("**Question:**", qa['question'])
                    st.write("**Answer:**", qa['answer'])
        
        if st.button("💡 Generate Automatic Insights"):
            with st.spinner("Generating insights..."):
                insights = st.session_state.agent.generate_insights()
            
            st.subheader("🔍 AI-Generated Insights")
            st.write(insights)
    
    with tab5:
        st.header("📋 Analysis Report")
        
        if st.button("📄 Generate Report"):
            with st.spinner("Preparing report..."):
                report = st.session_state.agent.export_analysis_report()
            
            st.markdown(report)
            
            st.download_button(
                label="⬇️ Download Report",
                data=report,
                file_name="data_analysis_report.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    try:
        import streamlit as st
        main()
    except ImportError:
        print("Streamlit not available. Running as standalone script.")
        print("To install Streamlit: pip install streamlit")
        print("To run the web app: streamlit run this_file.py")
        
        api_key = input("Enter your Together.ai API key: ")
        agent = DataAnalystAgent(api_key)
        
        file_path = input("Enter path to your data file: ")
        result = agent.load_document(file_path)
        
        if result['success']:
            print("✅ File loaded successfully!")
            insights = agent.generate_insights()
            print("\n🔍 AI-Generated Insights:")
            print(insights)
            
            while True:
                question = input("\nAsk a question (or 'quit' to exit): ")
                if question.lower() == 'quit':
                    break
                answer = agent.query_llm(question)
                print(f"\n🤖 Answer: {answer}")
        else:
            print(f"❌ Error loading file: {result.get('error')}")
