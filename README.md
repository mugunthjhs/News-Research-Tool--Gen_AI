# 📰 News Research Tool

![News Research Tool Output](https://github.com/mugunthjhs/News-Research-Tool--Gen_AI/blob/main/image_output.png)

This project is a web-based research tool that allows users to input news article URLs, process the content, and then query it using natural language questions. The tool uses OpenAI's language model for processing and generating insights from the provided articles.

## Features
- **Article Processing:** Input up to three article URLs, which are automatically processed and converted into vector embeddings using OpenAI's language model.
- **Query-Based Retrieval:** Users can ask questions related to the articles, and the tool provides responses along with the sources of the information.
- **Embeddings Storage:** The processed data is stored in a FAISS index, allowing for fast and efficient querying.
- **Streamlit Interface:** A simple and interactive interface built using Streamlit.

## How to Use

1. **Enter your OpenAI API key**:  
   In the sidebar of the Streamlit app, enter your OpenAI API key in the provided input field. This is essential for the tool to process and generate insights.

2. **Input News Article URLs**:  
   Enter up to three URLs of news articles you want to process. These articles will be fetched and split into smaller sections for analysis.

3. **Process the URLs**:  
   After entering the URLs, click the "Process URLs" button to begin fetching, splitting, and embedding the content. This step may take a few moments.

4. **Ask Questions**:  
   Once the articles are processed, you can ask any question related to the content by typing it in the input box at the bottom of the interface. The tool will retrieve relevant answers from the processed data.

5. **View the Results**:  
   The answer to your question, along with the sources from the articles, will be displayed in the main output section of the app.
