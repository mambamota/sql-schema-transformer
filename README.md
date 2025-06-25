# SQL Schema Transformer App

A Streamlit web app to help you transform SQL queries from an old table schema to a new one, using Excel files for schema definitions. Designed for easy deployment on Hugging Face Spaces and sharing with colleagues.

## Features
- Upload old and new schema Excel files (with sheets ending in `_c`)
- Automatically extract and map fields (using the 'Column Name' for SQL)
- Paste an old SQL query and transform it to match the new schema
- See a summary of column mappings
- View differences between old and new queries (side-by-side, with highlighting and line numbers)
- Download the transformed query

## Setup & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd "app composer ajust query"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app locally**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - Go to [http://localhost:8501](http://localhost:8501)

5. **How to use**
   - Upload your old and new schema Excel files
   - Select the relevant tables/sheets
   - Paste your old SQL query
   - Click "Transform Query" to see the new query and differences
   - Download the transformed query if needed

## Deploy on Hugging Face Spaces

1. Push this folder (with `app.py` and `requirements.txt`) to a GitHub repository.
2. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space.
3. Choose **Streamlit** as the SDK and connect your GitHub repo.
4. Set the Space's working directory to `app composer ajust query` if needed.
5. The app will be automatically built and deployed!

## Folder Structure
```
app composer ajust query/
├── app.py
├── requirements.txt
├── README.md
└── ...
```

## License
MIT 