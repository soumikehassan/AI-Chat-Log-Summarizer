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
    

    def preprocess_text(self, text): # Preprocess text for keyword extraction.
        
      
        # Lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation and numbers
        tokens = [token for token in tokens if token not in string.punctuation and not token.isdigit()]
        
        # Remove stop words and short words
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize tokens
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    def extract_keywords_simple(self, top_n=5): #Extract keywords using simple frequency counting.
        

        # Combine all messages
        all_text = ' '.join(self.user_messages + self.ai_messages)
        
        # Preprocess text
        tokens = self.preprocess_text(all_text)
        
        # Count word frequencies
        word_freq = Counter(tokens)
        
        # Get top keywords
        top_keywords = [word for word, _ in word_freq.most_common(top_n)]
        
        return top_keywords
    def extract_keywords_tfidf(self, top_n=5):
        """
        Extract keywords using TF-IDF.
        
        Args:
            top_n (int): Number of top keywords to extract
            
        Returns:
            list: List of top keywords
        """
        # Combine user messages and AI messages separately
        user_text = ' '.join(self.user_messages)
        ai_text = ' '.join(self.ai_messages)
        
        # Preprocess
        user_tokens = self.preprocess_text(user_text)
        ai_tokens = self.preprocess_text(ai_text)
        
        # Rejoin tokens
        user_text = ' '.join(user_tokens)
        ai_text = ' '.join(ai_tokens)
        
        # Create corpus
        corpus = [user_text, ai_text]
        
        # Apply TF-IDF
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Sum TF-IDF scores for each term across both documents
        tfidf_scores = {}
        for col in range(tfidf_matrix.shape[1]):
            term = feature_names[col]
            score = sum(tfidf_matrix[i, col] for i in range(tfidf_matrix.shape[0]))
            tfidf_scores[term] = score
        
        # Sort by score
        sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top keywords
        top_keywords = [term for term, _ in sorted_terms[:top_n]]
        
        return top_keywords
    
    def determine_conversation_nature(self, keywords): # Determine the nature of the conversation based on keywords.
        
        
        # This is a simple implementation and could be made more sophisticated
        if not keywords:
            return "The conversation topic could not be determined."
        
        # Join keywords for a simple description
        return f"The conversation was mainly about {', '.join(keywords)}."
    
    def generate_summary(self, use_tfidf=True): #Generate a summary of the chat log.
        
            
        
        # Get message statistics
        stats = self.get_message_stats()
        
        # Extract keywords
        if use_tfidf:
            keywords = self.extract_keywords_tfidf()
        else:
            keywords = self.extract_keywords_simple()
        
        # Determine conversation nature
        conversation_nature = self.determine_conversation_nature(keywords)
        
        # Generate summary
        summary = f"""Summary:
- The conversation had {stats['exchanges']} exchanges ({stats['total_messages']} total messages).
- The user sent {stats['user_messages']} messages and the AI sent {stats['ai_messages']} messages.
- {conversation_nature}
- Most common keywords: {', '.join(keywords)}.
"""
        return summary

def process_single_file(file_path, use_tfidf=True): #Process a single chat log file.
    
    
    summarizer = ChatLogSummarizer()
    if summarizer.parse_chat_log(file_path):
        return summarizer.generate_summary(use_tfidf)
    else:
        return "Failed to parse the chat log."

def process_directory(directory_path, use_tfidf=True): # Process all .txt files in a directory.
    
    
    results = {}
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            results[filename] = process_single_file(file_path, use_tfidf)
    
    return results

def main():
    """Main function to demonstrate usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Summarize AI chat logs.')
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('--no-tfidf', action='store_true', help='Use simple frequency counting instead of TF-IDF')
    
    args = parser.parse_args()
    use_tfidf = not args.no_tfidf
    
    if os.path.isdir(args.input):
        results = process_directory(args.input, use_tfidf)
        for filename, summary in results.items():
            print(f"\n=== Summary for {filename} ===")
            print(summary)
    else:
        summary = process_single_file(args.input, use_tfidf)
        print(summary)

if __name__ == "__main__":
    main()