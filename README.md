# Resume Relevance Checker

A Streamlit-based web application that analyzes resume relevance against job descriptions using AI-powered scoring and provides personalized improvement suggestions.

## Features

- **Document Processing**: Extract text from PDF and DOCX files
- **Skill Extraction**: Automatically identify must-have and good-to-have skills from job descriptions
- **Hybrid Scoring**: Combines keyword matching (hard) and semantic similarity (soft) scoring
- **AI Feedback**: Generate personalized improvement suggestions using Grok API
- **Interactive UI**: User-friendly Streamlit interface with real-time analysis
- **Export Results**: Save results as PDF files
## Project Structure

```
ResumeAI/
├── .cursor_tasks.json          # Task management file
├── .env                        # Environment variables (create from .env.example)
├── streamlit_app.py            # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── scoring/                    # Core scoring modules
│   ├── __init__.py
│   ├── parser.py              # Document text extraction
│   ├── skill_extractor.py     # Skill extraction from JDs
│   ├── scoring.py             # Hard/soft scoring logic
│   ├── embeddings.py          # Semantic similarity
│   ├── grok_client.py         # Grok API client
│   └── feedback.py            # AI feedback generation
├── data/                       # Data directories
│   ├── resumes/               # Uploaded resume files
│   ├── jds/                   # Uploaded job descriptions
│   ├── extracted_texts/       # Extracted text files
│   └── results/               # Analysis results
└── notebooks/                  # Jupyter notebooks for analysis
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ResumeAI
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your Grok API credentials
   ```

5. **Download additional models** (optional):
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   python -c "import spacy; spacy.cli.download('en_core_web_sm')"
   ```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Grok API Configuration
GROK_API_URL=https://api.grok.example/v1
GROK_API_KEY=sk-your-grok-api-key-here
```

### Grok API Setup

1. Sign up for Grok API access
2. Get your API key and base URL
3. Update the `.env` file with your credentials
4. The app will gracefully fallback to local methods if the API is unavailable

## Usage

### Running the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`.

### Basic Workflow

1. **Upload Job Description**:
   - Upload a PDF file or paste text directly
   - The app will automatically extract skills

2. **Upload Resumes**:
   - Upload multiple PDF or DOCX resume files
   - Files will be processed automatically

3. **Configure Scoring**:
   - Adjust hard/soft match weights using sliders
   - Default: 60% hard match, 40% soft match

4. **Analyze**:
   - Click "Analyze Resumes" to process all files
   - View results in the interactive table

5. **Export Results**:
   - Download results as CSV or JSON
   - Access detailed feedback for each resume

### Scoring Algorithm

The application uses a hybrid scoring approach:

1. **Hard Match (Keyword)**: 
   - Exact and fuzzy keyword matching
   - Percentage of must-have skills found
   - Weight: 60% (configurable)

2. **Soft Match (Semantic)**:
   - Embedding-based similarity
   - Cosine similarity between resume and JD
   - Weight: 40% (configurable)

3. **Final Score**:
   ```
   Final Score = (Hard Weight × Hard %) + (Soft Weight × Soft %)
   ```

4. **Verdict**:
   - High: ≥75%
   - Medium: 45-74%
   - Low: <45%

## API Integration

### Grok API

The application integrates with Grok API for:
- Skill extraction from job descriptions
- Semantic similarity calculations
- Personalized feedback generation

### Fallback Mechanisms

If Grok API is unavailable, the app uses:
- Rule-based skill extraction
- TF-IDF similarity for semantic matching
- Template-based feedback generation

## Development

### Real-time Internet Fetching (optional)

The project includes a lightweight internet fetching helper (`scoring/internet_tools.py`) that can augment Grok prompts with realtime web content. This is optional and disabled by default.

How to enable:

1. Install optional dependencies:

```bash
pip install langchain beautifulsoup4
```

2. Use `GrokClient.generate_with_online_context(prompt, urls=[...])` to fetch and include short summaries from the provided URLs before calling the Grok API.

Notes:
- Network calls may increase latency. Failures are handled gracefully and will fallback to local generation.
- Do not pass sensitive or private URLs unless you understand the privacy implications of sending them to an external API.


### Task Management

The application uses `.cursor_tasks.json` for task tracking:

```json
{
  "current_version": "v1",
  "tasks": [
    {
      "id": "task-01",
      "title": "Task description",
      "status": "done|in_progress|todo|blocked",
      "assigned_to": "cursor",
      "started_at": "2025-01-27T10:00:00Z",
      "finished_at": "2025-01-27T10:10:00Z",
      "outputs": ["file1.py", "file2.py"],
      "notes": "Task notes"
    }
  ],
  "last_updated": "2025-01-27T10:10:00Z"
}
```

### Adding New Features

1. Create a new module in the `scoring/` directory
2. Update the main Streamlit app to integrate the feature
3. Add tests in the `notebooks/` directory
4. Update the task management file

### Testing

Run the evaluation notebook:
```bash
jupyter notebook notebooks/eval.ipynb
```

## Troubleshooting

### Common Issues

1. **PDF Extraction Fails**:
   - Ensure PyMuPDF and pdfplumber are installed
   - Check file permissions and format

2. **Embedding Errors**:
   - Verify sentence-transformers installation
   - Check available memory for model loading

3. **Grok API Issues**:
   - Verify API credentials in `.env`
   - Check network connectivity
   - App will fallback to local methods

4. **Memory Issues**:
   - Reduce batch size for large files
   - Use smaller embedding models
   - Clear cache regularly

### Performance Optimization

- Use SSD storage for better I/O performance
- Increase available RAM for large document processing
- Consider using GPU for embedding calculations
- Implement caching for repeated operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the task management file for known issues
3. Create an issue in the repository
4. Contact the development team

## Changelog

### v1.0.0
- Initial release
- Basic resume scoring functionality
- Grok API integration
- Streamlit UI
- Export capabilities
- Task management system
