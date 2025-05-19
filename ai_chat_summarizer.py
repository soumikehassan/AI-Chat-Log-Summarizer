#!/usr/bin/env python3
"""
AI Chat Log Summarizer
"""

import os
import re
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class ChatLogSummarizer: #A class to parse and summarize AI chat logs.
    
    def __init__(self): #Initialize the summarizer with empty messages and stop words.
        self.user_messages = []
        self.ai_messages = []
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['would', 'could', 'should', 'using', 'used', 'use'])
        self.lemmatizer = WordNetLemmatizer()
        
    def parse_chat_log(self, file_path): #Parse a chat log file and separate user and AI messages.
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Split by speaker prefixes
            user_pattern = r'User: (.*?)(?=AI:|$)'
            ai_pattern = r'AI: (.*?)(?=User:|$)'
            
            # Find all matches using regex
            user_matches = re.findall(user_pattern, content, re.DOTALL)
            ai_matches = re.findall(ai_pattern, content, re.DOTALL)
            
            # Clean the messages (remove extra whitespace)
            self.user_messages = [msg.strip() for msg in user_matches]
            self.ai_messages = [msg.strip() for msg in ai_matches]
            
            return True
        
        except Exception as e:
            print(f"Error parsing chat log: {e}")
            return False
    


    def get_message_stats(self): #Calculate message statistics.
        
        stats = {
            'total_messages': len(self.user_messages) + len(self.ai_messages),
            'user_messages': len(self.user_messages),
            'ai_messages': len(self.ai_messages),
            'exchanges': min(len(self.user_messages), len(self.ai_messages))
        }
        return stats