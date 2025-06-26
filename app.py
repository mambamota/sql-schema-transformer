import streamlit as st
import pandas as pd
from io import BytesIO
from fuzzywuzzy import process
import difflib
from difflib import SequenceMatcher
import sqlparse
import re

st.set_page_config(page_title="SQL Schema Transformer", layout="wide")
st.title("SQL Query Schema Transformer (Auto Table/Column Detection)")

# --- Helper Functions ---
def extract_schema_from_excel(file, min_fields=1):
    """
    Extracts schema info from all sheets ending with '_c'.
    For each sheet, finds the table name in the row where column A is 'Table Name',
    scanning all columns in that row (except column A) for the first non-empty cell.
    Extracts all column names from the 'Column Name' column in the 'Custom Fields' section.
    Returns a dict: {object_name: {'table_name': ..., 'fields': [...]}}
    """
    xls = pd.ExcelFile(file)
    schema = {}
    for sheet in xls.sheet_names:
        if not sheet.endswith('_c'):
            continue
        df = xls.parse(sheet, header=None)
        # Find the table name: row where col 0 is 'Table Name', get first non-empty cell in that row (except col 0)
        table_name_row = df.index[df.iloc[:, 0].astype(str).str.strip() == 'Table Name']
        if len(table_name_row) == 0:
            continue
        row_idx = table_name_row[0]
        table_name = None
        for col_idx in range(1, df.shape[1]):
            cell = df.iloc[row_idx, col_idx]
            if pd.notnull(cell) and str(cell).strip():
                table_name = str(cell).strip()
                break
        if not table_name:
            continue
        # Find the row where the first column is exactly 'Custom Fields'
        custom_fields_row_idx = df.index[df.iloc[:, 0].astype(str).str.strip() == 'Custom Fields']
        if len(custom_fields_row_idx) == 0:
            continue
        header_idx = custom_fields_row_idx[0] + 1
        if header_idx >= len(df):
            continue
        header = [str(h).strip() for h in df.iloc[header_idx].fillna('').astype(str).tolist()]
        # Extract field info from the next rows (stop at first empty row)
        fields = []
        for i in range(header_idx + 1, len(df)):
            row = df.iloc[i]
            if row.isnull().all() or (row.astype(str) == '').all():
                break
            field_info = {header[j]: str(row.iloc[j]).strip() for j in range(min(len(header), len(row)))}
            col_name = field_info.get('Column Name')
            if col_name is not None and col_name.strip():  # Only add if 'Column Name' is not empty
                # Trim all keys and values
                field_info = {k.strip(): v.strip() for k, v in field_info.items()}
                fields.append(field_info)
        if len(fields) >= min_fields:
            schema[sheet.strip()] = {'table_name': table_name.strip(), 'fields': fields}
    return schema

def build_schema_dict(schema):
    """
    Returns dict: {table: set(columns)} for easier lookup.
    Trims whitespace from table and column names.
    """
    table_columns = {}
    for table, fields in schema.items():
        columns = set()
        for f in fields:
            col = f.get('Column Name')
            if col:
                columns.add(col.strip())
        table_columns[table.strip()] = columns
    return table_columns

def extract_tables_and_columns_from_query(query):
    """
    Use sqlparse and regex to extract all table and column names from the query.
    Returns: (set of tables, set of columns)
    """
    # Use sqlparse to get identifiers
    parsed = sqlparse.parse(query)
    tables = set()
    columns = set()
    # Regex for table.column or just column
    table_col_pattern = re.compile(r'([\w]+)\.([\w]+)')
    col_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\b')
    for stmt in parsed:
        for token in stmt.tokens:
            if token.ttype is None and hasattr(token, 'get_real_name'):
                name = token.get_real_name()
                if name:
                    tables.add(name)
            # Find table.column patterns
            for match in table_col_pattern.finditer(str(token)):
                tables.add(match.group(1))
                columns.add(match.group(2))
            # Find standalone columns (may include SQL keywords, filter later)
            for match in col_pattern.finditer(str(token)):
                columns.add(match.group(1))
    # Remove SQL keywords from columns
    sql_keywords = set([kw.upper() for kw in sqlparse.keywords.KEYWORDS.keys()])
    columns = set([c for c in columns if c.upper() not in sql_keywords])
    return tables, columns

def map_tables(old_tables, new_tables):
    """
    Map old table names to new table names using fuzzy matching.
    Returns dict: {old_table: new_table}
    """
    mapping = {}
    for old in old_tables:
        match, score = process.extractOne(old, list(new_tables))
        mapping[old] = match if score > 80 else old
    return mapping

def map_columns_for_table(old_fields, new_fields):
    """
    Map old columns to new columns using fuzzy matching across 'Column Name', 'Name', and 'Display Name'.
    old_fields and new_fields are lists of dicts (from schema extraction).
    Returns dict: {old_col: new_col}
    """
    mapping = {}
    # Build a list of (col_name, all_identifiers) for new fields
    new_col_candidates = []
    for nf in new_fields:
        col_name = nf.get('Column Name', '').strip()
        name = nf.get('Name', '').strip()
        display_name = nf.get('Display Name', '').strip()
        all_ids = [col_name, name, display_name]
        new_col_candidates.append((col_name, all_ids))
    for of in old_fields:
        old_col = of.get('Column Name', '').strip()
        old_name = of.get('Name', '').strip()
        old_display_name = of.get('Display Name', '').strip()
        old_ids = [old_col, old_name, old_display_name]
        best_score = 0
        best_new_col = old_col
        for new_col, new_ids in new_col_candidates:
            for oid in old_ids:
                for nid in new_ids:
                    if oid and nid:
                        score = process.extractOne(oid, [nid])[1]
                        if score > best_score:
                            best_score = score
                            best_new_col = new_col
        mapping[old_col] = best_new_col if best_score > 80 else old_col
    return mapping

def find_best_object_for_table(table_name, columns_in_query, schema):
    candidates = []
    for object_name, obj_info in schema.items():
        if obj_info['table_name'] == table_name:
            object_columns = {f.get('Column Name') for f in obj_info['fields']}
            overlap = len(set(columns_in_query) & object_columns)
            candidates.append((object_name, overlap))
    candidates.sort(key=lambda x: x[1], reverse=True)
    if candidates and candidates[0][1] > 0:
        return candidates[0][0]  # object_name with most overlap
    elif candidates:
        return candidates[0][0]  # fallback: just pick the first
    else:
        return None

def build_full_mapping_filtered(detected_table_columns, old_schema, new_schema):
    relevant_old_tables = set(detected_table_columns.keys())
    # Build a mapping from table_name to columns used in query
    table_map = {}
    column_maps = {}
    relevant_objects = set()
    for table_name in relevant_old_tables:
        columns_in_query = detected_table_columns[table_name]
        old_object = find_best_object_for_table(table_name, columns_in_query, old_schema)
        # Prefer to map to the same object name in new schema if it exists
        new_object = None
        if old_object and old_object in new_schema:
            new_object = old_object
        else:
            new_object = find_best_object_for_table(table_name, columns_in_query, new_schema)
        if not old_object or not new_object:
            continue
        # Map old table name to new table name using object name as the link
        old_table = old_schema[old_object]['table_name']
        new_table = new_schema[new_object]['table_name']
        table_map[old_table] = new_table
        relevant_objects.add(old_table)
        # Get all old fields (dicts) for this object
        old_fields = [f for f in old_schema[old_object]['fields'] if f.get('Column Name') and f.get('Column Name') in columns_in_query]
        new_fields = [f for f in new_schema[new_object]['fields'] if f.get('Column Name')]
        column_maps[old_table] = map_columns_for_table(old_fields, new_fields)
    return table_map, column_maps, relevant_objects

def transform_query_full(query, table_map, column_maps):
    """
    Replace old table and column names with new ones in the query.
    """
    # Replace table names first
    for old_table, new_table in table_map.items():
        if old_table != new_table:
            query = re.sub(rf'\b{re.escape(old_table)}\b', new_table, query)
    # Replace columns (qualified and unqualified)
    for old_table, col_map in column_maps.items():
        for old_col, new_col in col_map.items():
            if old_col != new_col:
                # Replace qualified: old_table.old_col
                query = re.sub(rf'\b{re.escape(old_table)}\.{re.escape(old_col)}\b', f'{table_map[old_table]}.{new_col}', query)
                # Replace unqualified: old_col (only if not part of another word)
                query = re.sub(rf'\b{re.escape(old_col)}\b', new_col, query)
    return query

def extract_table_aliases_and_columns(query, schema):
    """
    Extracts table/alias pairs and maps columns to their real tables from a SQL query, only for real tables in the schema (ignores CTEs).
    Returns: {real_table: set(columns_used)}
    """
    # Find all FROM/JOIN ... <table> <alias> patterns
    table_alias_pattern = re.compile(r'(FROM|JOIN)\s+([\w]+)\s+([\w]+)', re.IGNORECASE)
    alias_to_table = {}
    real_table_names = {obj['table_name'] for obj in schema.values()}
    for match in table_alias_pattern.finditer(query):
        real_table = match.group(2).strip()
        alias = match.group(3).strip()
        # Only keep if real_table is in schema (ignore CTEs)
        if real_table in real_table_names:
            alias_to_table[alias] = real_table
            alias_to_table[real_table] = real_table  # allow direct table usage too
    # Also handle FROM ... <table> (no alias)
    table_pattern = re.compile(r'(FROM|JOIN)\s+([\w]+)(?!\s+AS)', re.IGNORECASE)
    for match in table_pattern.finditer(query):
        real_table = match.group(2).strip()
        if real_table in real_table_names:
            alias_to_table[real_table] = real_table
    # Find all qualified column references (alias.column)
    qualified_col_pattern = re.compile(r'([\w]+)\.([\w]+)')
    table_columns = {}
    for match in qualified_col_pattern.finditer(query):
        alias = match.group(1).strip()
        col = match.group(2).strip()
        real_table = alias_to_table.get(alias)
        if real_table:
            table_columns.setdefault(real_table, set()).add(col)
    return {k.strip(): {c.strip() for c in v} for k, v in table_columns.items()}

# --- Streamlit UI ---
st.sidebar.header("1. Upload Schema Files")
old_file = st.sidebar.file_uploader("Upload OLD schema Excel file", type=["xlsx"])
new_file = st.sidebar.file_uploader("Upload NEW schema Excel file", type=["xlsx"])

if old_file and new_file:
    st.sidebar.success("Both files uploaded!")
    old_schema = extract_schema_from_excel(old_file)
    new_schema = extract_schema_from_excel(new_file)
    if not old_schema or not new_schema:
        st.error("Could not extract schema from one or both files. Check file format.")
    else:
        st.write("### Old Schema Tables:", list(old_schema.keys()))
        st.write("### New Schema Tables:", list(new_schema.keys()))
        st.header("2. Paste Old SQL Query")
        old_query = st.text_area("Paste your old SQL query here:", height=200)
        if old_query.strip():
            # Detect tables/columns used in query (robust alias-aware, only real tables)
            detected_table_columns = extract_table_aliases_and_columns(old_query, old_schema)
            # Only keep tables that are present in the schema (real tables)
            real_tables = {obj['table_name'] for obj in old_schema.values()}
            filtered_detected_table_columns = {k: v for k, v in detected_table_columns.items() if k in real_tables}
            st.write("#### Detected Tables and Columns (by robust alias extraction, filtered to real tables):")
            st.json(filtered_detected_table_columns)
            # Build mapping using only real tables
            table_map, column_maps, relevant_old_tables = build_full_mapping_filtered(filtered_detected_table_columns, old_schema, new_schema)
            st.write("#### Table Mapping:")
            st.json(table_map)
            st.write("#### Column Mapping (per table):")
            st.json(column_maps)
            # Show mapping summary
            for old_table in relevant_old_tables:
                st.write(f"##### Mapping for Table: {old_table} → {table_map[old_table]}")
                mapping_df = pd.DataFrame({
                    'Old Column Name': list(column_maps[old_table].keys()),
                    'New Column Name': list(column_maps[old_table].values())
                })
                st.dataframe(mapping_df)
            # Transform query
            if st.button("Transform Query"):
                new_query = transform_query_full(old_query, table_map, column_maps)
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