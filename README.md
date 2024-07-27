
Legal Document Analysis
Easwar Hariharan | 12th June 2024
INTRODUCTION
This progress report outlines the current status of the Legal Document Analysis Project, detailing changes, accomplished tasks, and future plans. The project is structured using agile methodologies, with the current reporting period marking the end of Semester 1.
CHANGES
Hugging Face Integration: Initial prototypes for basic analysis functionalities were developed using various LLMs hosted by Hugging Face instead of GPT-3 due to better suitability for this project.

Functionality Implementation:
Summarization: Implemented using Falconsai/text_summarization, a T5 transformer-based pre-trained model.
Sentiment Analysis: Implemented using siebert/sentiment-roberta-large-english, a BERT-based pre-trained model. Fine-tuning of this model was implemented.
Named Entity Recognition: Implemented using dslim/bert-large-NER, also a BERT-based pre-trained model.
ACCOMPLISHED TASKS
Tokenization and Dataset Preparation: The dataset from the GLUE benchmark (SST-2) was tokenized and prepared for model training and evaluation.
Model Training and Evaluation: A sentiment analysis model (siebert/sentiment-roberta-large-english) was fine-tuned and evaluated.
Pipelines Setup: Pipelines for summarization, sentiment analysis, and NER were set up and tested on sample texts from the dataset.
FUTURE PLANS
Advanced Techniques and Model Development
Implement Advanced Techniques: Refine the summarization, sentiment analysis, and NER functionalities to more advanced functionalities, possibly more general text classification.
Model evaluation, selection and fine-tuning: Evaluate and select models according to new requirements and further fine-tune them to the task at hand.
Testing and Validation
Testing Protocols: Develop and execute testing protocols to ensure the reliability and accuracy of the system.
Validation Techniques: Implement validation techniques to verify the performance of the system on various legal document datasets.
