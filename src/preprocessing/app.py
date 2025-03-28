import streamlit as st
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Data Upload & Explorer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stDataFrame {
        border: 1px solid #444;
        border-radius: 5px;
    }
    .stExpander {
        background: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
    }
    .highlight {
        background-color: #fffd80;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("Data Preprocessing Tool")
    
    # File upload section
    with st.container(border=True):
        st.subheader("1. Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed"
        )

    if uploaded_file:
        # Read data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Store in session state
            st.session_state.df = df

            # Show success and basic info
            st.success(f"✅ Successfully loaded {len(df)} rows × {len(df.columns)} columns")
            
            # Display raw data
            with st.expander("🔍 View Raw Data", expanded=False):
                st.dataframe(
                    df.style.highlight_null(color='#ffcccb'), 
                    height=300,
                    use_container_width=True
                )

            # Column information
            with st.expander("📋 Column Information", expanded=True):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes,
                    'Missing Values': df.isna().sum(),
                    'Unique Values': df.nunique(),
                    'Sample Value': df.iloc[0].values
                })
                
                st.dataframe(
                    col_info.style.applymap(
                        lambda x: 'background-color: #e6f3ff' if x == 'object' else '',
                        subset=['Type']
                    ),
                    use_container_width=True
                )

                # Quick stats
                st.markdown(f"""
                - **Total Columns:** {len(df.columns)}
                - **Total Rows:** {len(df)}
                - **Missing Values:** {df.isna().sum().sum()}
                - **Duplicate Rows:** {df.duplicated().sum()}
                """)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
