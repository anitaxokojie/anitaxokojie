# Hi, I'm Anita

I build NLP systems and work with large-scale data. I have a BA in Modern Languages & Linguistics from Warwick and an MA in Computational Linguistics from Goldsmiths. I speak English, Spanish, and French, which comes in handy for multilingual NLP work.

Most of my projects focus on making language models more explainable and figuring out how linguistic theory can improve ML performance. I care about building things that actually work and understanding why they work (or don't).

## Technical Skills

**Programming & Data**  
Python (pandas, NumPy), SQL

**Natural Language Processing**  
Transformers (BERT, DistilBERT, Sentence-Transformers), spaCy, NLTK, LLMs (Google Gemini), Regex, Corpus Linguistics, Semantic/Syntactic Analysis, Multi-Head Attention, Knowledge Distillation

**Machine Learning**  
scikit-learn, XGBoost, PyTorch, TensorFlow, K-means, ALS Collaborative Filtering, Feature Engineering, Data Preprocessing, Cross-Validation, Hyperparameter Optimization, LIME (Explainable AI)

**Big Data & Distributed Systems**  
Apache Spark (PySpark, MLlib), Hadoop MapReduce (HDFS), Apache Mahout, Distributed Computing Principles

**Data Visualization**  
Matplotlib, Seaborn

**Research & Methodologies**  
Systematic Literature Review, Contrastive Linguistics, Error Analysis, Fairness & Bias Analysis, Usability Testing, Graph-based Algorithms (PageRank, NetworkX)

## Projects

### Legal Document Relevance Assessment (Field Project)
Built a multi-stage information retrieval system for legal e-discovery that breaks down hundreds of thousands of documents into "atomic units" to make review more efficient. Combined Google Gemini 1.5-flash for LLM-based thematic extraction with a custom PageRank algorithm and got a mean F1 score of 0.840 (LLM + TF-IDF) for relevance classification.

The interesting part: optimized API calls and semantic similarity calculations through intelligent caching, which cut costs by 73% while still completing 99%+ of evaluations. Also identified significant cross-domain performance degradation of 47.6%, which was a useful finding for understanding the system's limitations.

**Technologies:** Python, Google Gemini 1.5-flash API, NetworkX, scikit-learn, BM25

### Multilingual Extractive Summarization
Created a BERT-based extractive summarization system for 1,000+ TED Talk transcripts in English and Spanish. Achieved ROUGE-1 F1 of 0.3078 and BLEU-1 of 0.2281. Implemented language-specific preprocessing and domain-specific boosting, which led to a 40% speed improvement and 27% enhanced topic fidelity.

Ran a fairness analysis and found some biases I wasn't expecting: the system showed bias towards technology content (+20.5% length) and underrepresentation of society-focused talks (-19% length). These findings informed my thinking about ethical AI development.

**Technologies:** Python, Transformers, spaCy, PageRank, NLTK, LIME

### Fake News Detection with Hybrid ML
Designed and implemented a hybrid ML system that combines transformer embeddings with linguistic analysis for fake news detection, achieving 94.03% accuracy. Engineered a comprehensive feature extraction pipeline including readability, sentiment, and lexical diversity metrics, plus novel fusion methods to optimally integrate these signals.

Built interpretability modules using LIME for transparent feature visualization, which helped identify linguistic markers. Found that fake news shows more polarized sentiment (-0.257 vs -0.016 for real news), which makes sense but is interesting to see quantified.

**Technologies:** Python, PyTorch, Transformers, VADER, scikit-learn, LIME

### Big Data E-commerce Analytics
Developed a distributed big data analytics solution using Apache Spark for 500,000+ e-commerce records across 7 tables, demonstrating horizontal scalability. Implemented customer segmentation via K-means clustering (identifying High-Value Loyal, High-Value Occasional, Low-Engagement segments) and an ALS collaborative filtering recommendation system achieving 0.82 RMSE.

The analytics side identified Electronics as having the highest profit margin at 49.7% and pinpointed 27 high-selling products with critical inventory levels.

**Technologies:** PySpark, MLlib, Hadoop, SQL, Alternating Least Squares

## Professional Experience

**Online TED Translator** (March 2024 – Present)  
Volunteer to transcribe, translate, and edit subtitles for TED Talks using the CaptionHub platform, making diverse content globally accessible. Focus on high-quality linguistic output and adherence to subtitling standards, with strong attention to detail and cross-cultural communication.

**Researcher for Coventry Council Translation (CITU)** (January 2024 – March 2024)  
Conducted long-term research for Coventry Interpretation & Translation Unit to explore technological solutions for increasing service efficiency in booking and delivery. Investigated applications of machine translation to identify opportunities for business improvement and streamline operational processes.

**Internship • Comtec Translations • Leamington Spa, UK** (March 2024)  
Supported administrative tasks and identified language suppliers for various translation projects, contributing to operational workflows. Gained practical experience researching and utilizing Computer-Assisted Translation (CAT) tools, applying them to practice translations from Spanish and French into English.

## Community Involvement

**Member • Women in AI**  
Actively participate in initiatives promoting diversity, ethical considerations, and community building within the AI field.

**Participant • NLP Reading Group**  
Engage in critical analysis and discussions of cutting-edge NLP research papers, deepening theoretical understanding of models like Transformers and fostering continuous learning.

## Education

**MA Computational Linguistics** — University of Goldsmiths

**BA Modern Languages & Linguistics** — University of Warwick

## Contact

📧 anitaxokojie@gmail.com  
💼 [linkedin.com/in/anitaxo](http://www.linkedin.com/in/anitaxo)
