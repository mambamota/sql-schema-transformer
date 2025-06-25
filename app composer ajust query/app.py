import streamlit as st
import pandas as pd
from io import BytesIO
from fuzzywuzzy import process
import difflib
from difflib import SequenceMatcher
import sqlparse

st.set_page_config(page_title="SQL Schema Transformer", layout="wide")
st.title("SQL Query Schema Transformer")

# --- Helper Functions ---
def extract_schema_from_excel(file, min_fields=1):
    """
    Extracts schema info from all sheets ending with '_c'.
    Returns a dict: {sheet_name: [list of dicts, one per field, with keys from the header row]}
    """
    xls = pd.ExcelFile(file)
    schema = {}
    for sheet in xls.sheet_names:
        if not sheet.endswith('_c'):
            continue
        df = xls.parse(sheet, header=None)
        # Find the row where the first column is exactly 'Custom Fields'
        custom_fields_row_idx = df.index[df.iloc[:, 0].astype(str).str.strip() == 'Custom Fields']
        if len(custom_fields_row_idx) == 0:
            continue
        header_idx = custom_fields_row_idx[0] + 1
        if header_idx >= len(df):
            continue
        header = df.iloc[header_idx].fillna('').astype(str).str.strip().tolist()
        # Extract field info from the next rows (stop at first empty row)
        fields = []
        for i in range(header_idx + 1, len(df)):
            row = df.iloc[i]
            if row.isnull().all() or (row.astype(str) == '').all():
                break
            field_info = {header[j]: str(row.iloc[j]).strip() for j in range(min(len(header), len(row)))}
            if field_info.get(header[0]):  # Only add if first column (Name) is not empty
                fields.append(field_info)
        if len(fields) >= min_fields:
            schema[sheet] = fields
    return schema

def map_fields(old_fields, new_fields):
    """
    Map old fields to new fields using fuzzy matching.
    Returns a dict: {old_field: new_field}
    """
    mapping = {}
    for old in old_fields:
        match, score = process.extractOne(old, new_fields)
        mapping[old] = match if score > 80 else old  # Only map if high confidence
    return mapping

def transform_query(query, field_map, old_table, new_table):
    """
    Replace old field and table names with new ones in the query.
    """
    # Replace table name
    if old_table != new_table:
        query = query.replace(old_table, new_table)
    # Replace field names
    for old_field, new_field in field_map.items():
        if old_field != new_field:
            query = query.replace(old_field, new_field)
    return query

def extract_all_sheet_data(file):
    """
    Extracts all data from all sheets ending with '_c'.
    Returns a dict: {sheet_name: DataFrame}
    """
    xls = pd.ExcelFile(file)
    all_data = {}
    for sheet in xls.sheet_names:
        if sheet.endswith('_c'):
            df = xls.parse(sheet, header=None)
            all_data[sheet] = df
    return all_data

# --- Streamlit UI ---
st.sidebar.header("1. Upload Schema Files")
old_file = st.sidebar.file_uploader("Upload OLD schema Excel file", type=["xlsx"])
new_file = st.sidebar.file_uploader("Upload NEW schema Excel file", type=["xlsx"])

if old_file and new_file:
    st.sidebar.success("Both files uploaded!")
    old_schema = extract_schema_from_excel(old_file)
    new_schema = extract_schema_from_excel(new_file)
    old_all_data = extract_all_sheet_data(old_file)
    new_all_data = extract_all_sheet_data(new_file)
    if not old_schema or not new_schema:
        st.error("Could not extract schema from one or both files. Check file format.")
    else:
        st.write("### Old Schema Tables:", list(old_schema.keys()))
        st.write("### New Schema Tables:", list(new_schema.keys()))
        # Table selection
        old_table = st.selectbox("Select OLD table (sheet)", list(old_schema.keys()))
        new_table = st.selectbox("Select NEW table (sheet)", list(new_schema.keys()), index=0)
        # Toggle for all data or just parsed fields
        show_all_data = st.checkbox("Show ALL sheet data (not just Custom Fields)", value=False)
        if show_all_data:
            st.write(f"#### All Data for Old Table ({old_table}):")
            st.dataframe(old_all_data[old_table])
            st.write(f"#### All Data for New Table ({new_table}):")
            st.dataframe(new_all_data[new_table])
        else:
            st.write(f"#### Old Fields ({old_table}):")
            st.dataframe(pd.DataFrame(old_schema[old_table]))
            st.write(f"#### New Fields ({new_table}):")
            st.dataframe(pd.DataFrame(new_schema[new_table]))
        # Field mapping (use 'Column Name' for mapping)
        old_col_names = [f['Column Name'] for f in old_schema[old_table] if f.get('Column Name')]
        new_col_names = [f['Column Name'] for f in new_schema[new_table] if f.get('Column Name')]
        field_map = map_fields(old_col_names, new_col_names)
        st.write("#### Field Mapping (Column Name):")
        st.json(field_map)
        # Add summary table of mapped columns
        if field_map:
            st.write("#### Column Mapping Summary:")
            mapping_df = pd.DataFrame({
                'Old Column Name': list(field_map.keys()),
                'New Column Name': list(field_map.values())
            })
            st.dataframe(mapping_df)
        # SQL input
        st.header("2. Paste Old SQL Query")
        old_query = st.text_area("Paste your old SQL query here:", height=200)
        if st.button("Transform Query") and old_query.strip():
            new_query = transform_query(old_query, field_map, old_table, new_table)
            st.header("3. Transformed SQL Query")
            st.code(new_query, language="sql")
            st.download_button("Download New Query", new_query, file_name="transformed_query.sql")
            # Side-by-side full queries with perfect formatting and syntax highlighting
            st.header("5. Side-by-Side Full Queries (SQL Syntax Highlighting)")
            formatted_old_query = sqlparse.format(old_query, reindent=True, keyword_case='upper')
            formatted_new_query = sqlparse.format(new_query, reindent=True, keyword_case='upper')
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Old Query**")
                st.code(formatted_old_query, language="sql")
            with col2:
                st.markdown("**New Query**")
                st.code(formatted_new_query, language="sql")

            # Custom HTML block with line numbers and inline color highlighting for differences
            st.header("6. Side-by-Side Diff with Line Numbers and Highlighting")
            old_lines = formatted_old_query.splitlines()
            new_lines = formatted_new_query.splitlines()
            max_lines = max(len(old_lines), len(new_lines))
            html = '<div style="display: flex; gap: 32px;">'
            # Old Query Column
            html += '<div><b>Old Query</b><pre style="font-size:13px; background:#f8f8f8; padding:8px;">'
            for i in range(max_lines):
                old_line = old_lines[i] if i < len(old_lines) else ''
                new_line = new_lines[i] if i < len(new_lines) else ''
                matcher = SequenceMatcher(None, old_line, new_line)
                old_out = ''
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    text = old_line[i1:i2]
                    if tag == 'equal':
                        old_out += text
                    else:
                        old_out += f'<span style="background-color:#ffcccc">{text}</span>' if text else ''
                html += f'<span style="color:#888">{i+1:>3} </span>{old_out}\n'
            html += '</pre></div>'
            # New Query Column
            html += '<div><b>New Query</b><pre style="font-size:13px; background:#f8f8f8; padding:8px;">'
            for i in range(max_lines):
                old_line = old_lines[i] if i < len(old_lines) else ''
                new_line = new_lines[i] if i < len(new_lines) else ''
                matcher = SequenceMatcher(None, old_line, new_line)
                new_out = ''
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    text = new_line[j1:j2]
                    if tag == 'equal':
                        new_out += text
                    else:
                        new_out += f'<span style="background-color:#ccffcc">{text}</span>' if text else ''
                html += f'<span style="color:#888">{i+1:>3} </span>{new_out}\n'
            html += '</pre></div>'
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)

            # Show a summary of changed lines below
            st.header("7. Changed Lines Summary")
            diff = list(difflib.ndiff(old_lines, new_lines))
            old_idx = 0
            new_idx = 0
            diff_lines = []
            for line in diff:
                marker = line[:2]
                content = line[2:]
                if marker == '  ':
                    old_idx += 1
                    new_idx += 1
                elif marker == '- ':
                    diff_lines.append(f"Old {old_idx+1:>3} |     | - {content}")
                    old_idx += 1
                elif marker == '+ ':
                    diff_lines.append(f"    | New {new_idx+1:>3} | + {content}")
                    new_idx += 1
                elif marker == '? ':
                    diff_lines.append(f"    |     | ? {content}")
            if diff_lines:
                st.code('\n'.join(diff_lines), language="diff")
            else:
                st.info("No differences found between old and new query.")
else:
    st.info("Please upload both old and new schema Excel files to begin.") 