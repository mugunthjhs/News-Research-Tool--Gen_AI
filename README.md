# URL Insight Assistant

[**Access the App:  https://url-insight-assistant.streamlit.app/ (Streamlit Cloud Deployment)**]

A Streamlit application that allows you to extract content from URLs, index it, and ask questions about the information. Leverages Google's Gemini models (1.5 Flash) for powerful and efficient information retrieval.

## Features

*   **URL Processing:** Ingest and process content from one or more URLs.
*   **Content Indexing:** Creates a vector index of the URL content for efficient searching.
*   **Question Answering:** Ask questions about the content of the processed URLs and receive concise answers.
*   **Source Attribution:** Provides the source URLs used to answer questions.
*   **Streamlit Interface:** User-friendly interface for easy interaction.

## Getting Started

### Prerequisites

*   A Google Generative AI API key. You can obtain one from the [Google AI Studio](https://makersuite.google.com/).
*   Python 3.7+
*   Streamlit

### Installation

1.  Clone the repository:

    ```bash
    git clone [your-repository-url]  # Replace with your actual repository URL
    cd your-repository-directory
    ```

2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

    (Make sure you have a `requirements.txt` file that includes: `streamlit`, `langchain-google-genai`, `langchain-community`, `langchain`, `unstructured`)

### Usage

1.  Run the Streamlit application:

    ```bash
    streamlit run main.py
    ```

    (Assuming your main Streamlit file is named `main.py`)

2.  Enter your Google Generative AI API key in the sidebar.
3.  Enter the URLs you want to process in the sidebar.
4.  Click the "Process URLs" button.
5.  Once the URLs are processed, you can ask questions in the text input box.

## Notes

*   The vector index is stored locally in a directory named `vector_index`.
*   The application uses Gemini 1.5 Flash for question answering. Make sure your API key has access to this model.
*   Error Handling: The UI incorporates alerts for issues stemming from keys and incorrect data input.
