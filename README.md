
    
    ### What does this application do?
    This application processes and groups a list of keywords based on their semantic similarity. It uses the OpenAI API to generate embeddings and the DBSCAN algorithm to perform clustering.

    ### How to use this application?
    1. **Upload an Excel file** containing the text data to be analyzed.
    2. **Select the column** containing the text data in the Excel file.
    3. **Configure the parameters** in the sidebar:
        - Enter your OpenAI API key.
        - Choose the embedding and clustering models.
        - Set the `eps` and `min_samples` values for the DBSCAN algorithm.
        - Define the chunk size for the content.
    4. **Click "Run"** to start the processing.
    5. Review and download the results:
        - An Excel file of the processed data.
        - A DataFrame showing cluster similarity.
        - An interactive knowledge graph of the clusters.

    ### How to run this application locally?
    1. **Clone the repository** to your local machine:
        ```
        git clone https://github.com/your_username/your_repository.git
        ```
    2. **Navigate to the project directory**:
        ```
        cd your_repository
        ```
    3. **Create a virtual environment** and activate it:
        ```
        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
        ```
    4. **Install the required dependencies**:
        ```
        pip install -r requirements.txt
        ```
    5. **Run the Streamlit application in a terminal**:
        ```
        streamlit run cluster.py
        ```
    6. **Open your browser** and go to the provided URL to use the application.
