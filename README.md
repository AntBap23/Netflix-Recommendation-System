# 🎬 Netflix Recommendation System

This project is a **content-based recommendation system** designed to suggest movies and TV shows available on Netflix. It uses **Natural Language Processing (NLP)** and **similarity metrics** to recommend titles based on genres, descriptions, cast, and more.

## 🚀 Features

- **Content-based recommendations** using **TF-IDF vectorization** and **cosine similarity**.
- **Multi-feature similarity** including genre, description, director, cast, and country.
- **Interactive UI** using **Gradio**, allowing users to input a movie title and receive recommendations.
- **Preprocessing & Data Cleaning** to handle missing values, extract relevant features, and encode categorical variables.

## 📂 Dataset

The system is trained on the **Netflix Titles Dataset** (`netflix_titles.csv`), which includes:

- Movie and TV show titles
- Release year
- Genres
- Cast & director information
- Country of origin
- Description of the content
- Duration (for movies) & seasons (for TV shows)
- Age rating

## 🛠️ Installation

### Prerequisites

Ensure you have **Python 3.7+** installed along with the required dependencies.

### Clone the Repository

```bash
git clone https://github.com/AntBap23/Netflix-Recommendation-System.git
cd Netflix-Recommendation-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 🔧 Usage

### Running the Recommendation System

```bash
python app.py
```

This will launch a **Gradio UI**, allowing users to enter a movie title and receive recommendations.

### Example Usage

1. Enter a **Netflix title** (e.g., *Stranger Things*).
2. Click **Submit**.
3. View the **top 10 recommended titles** based on similarity metrics.

## 🏗️ How It Works

1. **Preprocessing & Feature Engineering:**
   - Handles missing values (e.g., fills missing descriptions, duration, etc.).
   - Extracts movie **duration** and **TV show seasons**.
   - Encodes **genres** and **countries** using **one-hot encoding**.

2. **Text Vectorization:**
   - **TF-IDF** vectorization for **description, director, and cast**.
   - Multi-hot encoding for **genres**.

3. **Similarity Computation:**
   - Uses **cosine similarity** to compare titles.
   - Generates a **final similarity score** by weighing features like genre, description, director, and cast.

4. **Recommendation Generation:**
   - Retrieves the **top 10 similar movies/TV shows**.

## 🌟 Example Output

```json
[
  {"title": "Breaking Bad", "release_year": "2008"},
  {"title": "El Camino: A Breaking Bad Movie", "release_year": "2019"},
  {"title": "Better Call Saul", "release_year": "2015"}
]
```

## 📊 Technologies Used

- **Python** (Data processing & model implementation)
- **Pandas & NumPy** (Data handling & preprocessing)
- **NLTK** (Text preprocessing & tokenization)
- **Scikit-learn** (Vectorization, similarity computation)
- **Gradio** (User Interface)

## 🖼️ Screenshots

### Gradio Interface

https://huggingface.co/spaces/Antbap23/Netflix-Recommendation-System

### Sample Recommendations Output

![Sample Output](screenshots/sample_output.png)

## 🤝 Contributing

Contributions are welcome! Feel free to open an **issue** or submit a **pull request** if you have improvements or bug fixes.

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

