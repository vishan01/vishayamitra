# Vishayamitra

Vishayamitra is a Streamlit application that provides various data analysis and visualization tools. It includes features like ChatBI, Pattern Identifier, Data Visualizer, Database Connector, and Data Editor.

## Features

*   **ChatBI**: Interact with your data using natural language queries. Powered by Google Generative AI and PandasAI. ([`pages/home.py`](streamlit/pages/home.py))
*   **Pattern Identifier**: Train machine learning models for regression, classification, anomaly detection, clustering, and time series analysis. ([`pages/pattern.py`](streamlit/pages/pattern.py))
*   **Data Visualizer**: Visualize data from CSV, Excel, or JSON files using Pygwalker. ([`pages/visualization.py`](streamlit/pages/visualization.py))
*   **Database Connector**: Connect to SQL databases and retrieve data. ([`pages/sqldata.py`](streamlit/pages/sqldata.py))
*   **Data Editor**: Edit datasets interactively, drop unwanted columns, or modify data. ([`pages/data_editor.py`](streamlit/pages/data_editor.py))

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/your-username/vishayamitra.git
    cd vishayamitra
    ```
2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```
3.  Set up secrets:

    *   Add your API keys to `streamlit/.streamlit/secrets.toml`. You can obtain the `PANDASAI_API_KEY` and `GOOGLE_API_KEY` from their respective platforms.

    ```toml
    [secrets]
    PANDASAI_API_KEY = "your_pandasai_api_key"
    GOOGLE_API_KEY = "your_google_api_key"
    ```
4.  Run the application:

    ```bash
    streamlit run streamlit/app.py
    ```

## Usage

1.  Upload your dataset in CSV, Excel, or JSON format using the file uploader in the app.
2.  Navigate through the sidebar to access different tools. The sidebar links are defined in [`app.py`](streamlit/app.py), [`pages/home.py`](streamlit/pages/home.py), [`pages/pattern.py`](streamlit/pages/pattern.py), [`pages/visualization.py`](streamlit/pages/visualization.py), and [`pages/sqldata.py`](streamlit/pages/sqldata.py).
3.  Use the respective interfaces for each tool:

    *   **ChatBI**: Ask questions about your data in the text area and click "Submit".
    *   **Pattern Identifier**: Select the task type, x and y values, and the model. Train the model and make predictions.
    *   **Data Visualizer**: Explore and visualize your data using the Pygwalker interface.
    *   **Database Connector**: Enter the database connection details and click "Connect" to view and interact with the data.
    *   **Data Editor**: Edit the data table or drop columns as needed.

## Dependencies

The project uses the following main libraries:

*   streamlit==1.35.0
*   pandasai==2.2.5
*   langchain-google-genai==1.0.5
*   pygwalker==0.4.8.10
*   scikit-learn==1.5.0

A complete list of dependencies can be found in [requirements.txt](requirements.txt).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on the GitHub repository.