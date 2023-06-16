
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import datasets
from datasets import Dataset
import pyarrow as pa
import pyarrow.dataset as ds
from transformers import BertTokenizer

nltk.download('stopwords')
nltk.download('punkt')    

def preprocessing_text(text):
    stop_words = set(stopwords.words('catalan'))
    text = re.sub('http[s]?://\S+', '', text)                       # Remove urls
    text  = re.sub('[\d\W_]+', ' ', text)                           # Remove numbers, emojis, simbols, ...
    tokens = word_tokenize(text.lower())                            # Convert to lowercase and Tokenize the text 
    tokens = [token for token in tokens if token not in stop_words] # Remove stopwords
    preprocessed_text = ' '.join(tokens)                            # Join the tokens back into a string
    return preprocessed_text

def loading_data(path, task):
    data = pd.read_csv(path, sep='\t')                                      # Read the data into a DataFrame
    data = pd.DataFrame({'text': data['text'], 'label': data[task]})        # Keep only necessary columns (text, label)
    if task == 'task1':
        data['label']  = data['label'].map({'sexist': 1, 'non-sexist': 0})  # Map labels into integers
    
    if task == 'task2':
        data['label']  = data['label'].map({'ideological-inequality': 1,    # Map labels into integers
                                            'stereotyping-dominance': 2,
                                            'objectification': 3,
                                            'sexual-violence': 4,
                                            'misogyny-non-sexual-violence': 5, 
                                            'non-sexist': 0}) 
        
    data['text']  = data['text'].apply(preprocessing_text)                  # Preprocess the text
    return data



# Load tokenizer and define length of the text sequence
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')



# Define a function that will tokenize the model, and will return the relevant inputs for the model
def tokenization(batched_text):
    return tokenizer(batched_text['text'], padding = True, truncation=True)

def formatting_data(df):
    data = Dataset(pa.Table.from_pandas(df))                                    # Convert DataFrame to Huggingface dataset
    data = data.map(tokenization, batched = True, batch_size = len(data))       # Apply the tokenization function to the dataset    
    data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])  # Set the format of the dataset to PyTorch
    return data
    

