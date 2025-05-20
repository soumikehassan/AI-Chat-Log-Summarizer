#!/usr/bin/env python3
"""
AI Chat Log Summarizer

This script reads chat logs between a user and an AI, parses them,
and generates a summary including message counts and frequently used keywords.
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

class ChatLogSummarizer:
    """A class to parse and summarize AI chat logs."""
    
    def __init__(self):
        """Initialize the summarizer with empty messages and stop words."""
        self.user_messages = []
        self.ai_messages = []
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['would', 'could', 'should', 'using', 'used', 'use'])
        self.lemmatizer = WordNetLemmatizer()
        
    def parse_chat_log(self, file_path):
        """
        Parse a chat log file and separate user and AI messages.
        
        Args:
            file_path (str): Path to the chat log file
            
        Returns:
            bool: True if parsing was successful, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Try different parsing strategies
            
            # Strategy 1: Common formats with various prefixes
            self.user_messages = []
            self.ai_messages = []
            
            # Define patterns for different prefixes
            user_patterns = [
                r'User: (.*?)(?=AI:|AI Assistant:|Assistant:|$)',
                r'Human: (.*?)(?=AI:|AI Assistant:|Assistant:|$)'
            ]
            
            ai_patterns = [
                r'AI: (.*?)(?=User:|Human:|$)',
                r'AI Assistant: (.*?)(?=User:|Human:|$)',
                r'Assistant: (.*?)(?=User:|Human:|$)'
            ]
            
            # Try to find matches using each pattern
            for pattern in user_patterns:
                user_matches = re.findall(pattern, content, re.DOTALL)
                if user_matches:
                    # Clean the messages (remove extra whitespace)
                    self.user_messages.extend([msg.strip() for msg in user_matches if msg.strip()])
            
            for pattern in ai_patterns:
                ai_matches = re.findall(pattern, content, re.DOTALL)
                if ai_matches:
                    # Clean the messages (remove extra whitespace)
                    self.ai_messages.extend([msg.strip() for msg in ai_matches if msg.strip()])
            
            # If no messages found, try alternative parsing method
            if not self.user_messages and not self.ai_messages:
                print(f"Standard parsing failed for {file_path}, trying alternative methods...")
                
                # Strategy 2: Line-by-line detection of speakers
                lines = content.split('\n')
                lines = [line.strip() for line in lines if line.strip()]
                
                # Try to infer the conversation structure
                # Look for patterns like "User:", "AI:", "Human:", "Assistant:", etc.
                user_indicators = ["user:", "human:", "customer:", "client:", "me:", "question:"]
                ai_indicators = ["ai:", "ai assistant:", "assistant:", "bot:", "system:", "chatbot:", "response:", "answer:"]
                
                current_speaker = None
                current_message = []
                
                for line in lines:
                    lower_line = line.lower()
                    
                    # Check if line indicates a new speaker
                    if any(lower_line.startswith(indicator) for indicator in user_indicators):
                        # Save previous message if exists
                        if current_speaker == "user" and current_message:
                            self.user_messages.append(' '.join(current_message))
                        elif current_speaker == "ai" and current_message:
                            self.ai_messages.append(' '.join(current_message))
                            
                        # Start new user message
                        current_speaker = "user"
                        # Extract the message part (after the indicator)
                        for indicator in user_indicators:
                            if lower_line.startswith(indicator):
                                message_start = len(indicator)
                                current_message = [line[message_start:].strip()]
                                break
                    
                    elif any(lower_line.startswith(indicator) for indicator in ai_indicators):
                        # Save previous message if exists
                        if current_speaker == "user" and current_message:
                            self.user_messages.append(' '.join(current_message))
                        elif current_speaker == "ai" and current_message:
                            self.ai_messages.append(' '.join(current_message))
                            
                        # Start new AI message
                        current_speaker = "ai"
                        # Extract the message part (after the indicator)
                        for indicator in ai_indicators:
                            if lower_line.startswith(indicator):
                                message_start = len(indicator)
                                current_message = [line[message_start:].strip()]
                                break
                    
                    # If no new speaker indicator, continue with current message
                    elif current_speaker:
                        current_message.append(line)
                
                # Add the last message
                if current_speaker == "user" and current_message:
                    self.user_messages.append(' '.join(current_message))
                elif current_speaker == "ai" and current_message:
                    self.ai_messages.append(' '.join(current_message))
                
                # Strategy 3: Try to extract any text that looks like a conversation
                if not self.user_messages and not self.ai_messages:
                    print(f"Alternative parsing also failed for {file_path}, trying generic extraction...")
                    
                    # Look for specific prefixes in each line
                    for line in lines:
                        line_lower = line.lower()
                        if line_lower.startswith(tuple(user_indicators)):
                            # This is a user message, extract content after the prefix
                            for prefix in user_indicators:
                                if line_lower.startswith(prefix):
                                    message = line[len(prefix):].strip()
                                    if message:
                                        self.user_messages.append(message)
                                    break
                        elif line_lower.startswith(tuple(ai_indicators)):
                            # This is an AI message, extract content after the prefix
                            for prefix in ai_indicators:
                                if line_lower.startswith(prefix):
                                    message = line[len(prefix):].strip()
                                    if message:
                                        self.ai_messages.append(message)
                                    break
                
                # Strategy 4: Assume alternating speakers if still no messages
                if not self.user_messages and not self.ai_messages:
                    print(f"All structured parsing methods failed for {file_path}, assuming alternating speakers...")
                    
                    # Assume alternating speakers, starting with user
                    is_user_turn = True
                    
                    for line in lines:
                        if line.strip():  # Skip empty lines
                            if is_user_turn:
                                self.user_messages.append(line.strip())
                            else:
                                self.ai_messages.append(line.strip())
                            is_user_turn = not is_user_turn
            
            # Display parsing stats
            print(f"Parsed {len(self.user_messages)} user messages and {len(self.ai_messages)} AI messages from {file_path}")
            
            return len(self.user_messages) > 0 or len(self.ai_messages) > 0
        
        except Exception as e:
            print(f"Error parsing chat log {file_path}: {e}")
            return False
    
    def get_message_stats(self):
        """
        Calculate message statistics.
        
        Returns:
            dict: Dictionary containing message counts
        """
        total_user = len(self.user_messages)
        total_ai = len(self.ai_messages)
        
        # Calculate exchanges (conversation turns)
        # An exchange is a user message followed by an AI response
        # If there are uneven numbers of messages, we count as many complete exchanges as possible
        exchanges = min(total_user, total_ai)
        
        # If we have just one type of message (all user or all AI),
        # calculate exchanges based on conversation structure
        if total_user > 0 and total_ai == 0:
            # Conversations with only user messages - consider each message an "exchange"
            # This is a fallback as real exchanges need both parties
            exchanges = 0
        elif total_user == 0 and total_ai > 0:
            # Conversations with only AI messages - consider each message an "exchange"
            # This is a fallback as real exchanges need both parties
            exchanges = 0
            
        stats = {
            'total_messages': total_user + total_ai,
            'user_messages': total_user,
            'ai_messages': total_ai,
            'exchanges': exchanges
        }
        return stats
    
    def preprocess_text(self, text):
        """
        Preprocess text for keyword extraction.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            list: List of preprocessed tokens
        """
        # Lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation and numbers
        tokens = [token for token in tokens if token not in string.punctuation and not token.isdigit()]
        
        # Remove stop words and short words
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize tokens
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def extract_keywords_simple(self, top_n=5):
        """
        Extract keywords using simple frequency counting.
        
        Args:
            top_n (int): Number of top keywords to extract
            
        Returns:
            list: List of top keywords
        """
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
        
        # If both document token lists are empty, fall back to simple method
        if not user_tokens and not ai_tokens:
            return self.extract_keywords_simple(top_n)
            
        # Rejoin tokens
        user_text = ' '.join(user_tokens) if user_tokens else "dummy_text"
        ai_text = ' '.join(ai_tokens) if ai_tokens else "dummy_text"
        
        # Create corpus
        corpus = [user_text, ai_text]
        
        try:
            # Apply TF-IDF
            vectorizer = TfidfVectorizer(max_features=100, min_df=1)
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Filter out any dummy text we added
            feature_names = [name for name in feature_names if name != "dummy_text"]
            
            if not feature_names:
                return self.extract_keywords_simple(top_n)
            
            # Sum TF-IDF scores for each term across both documents
            tfidf_scores = {}
            for col in range(tfidf_matrix.shape[1]):
                term = vectorizer.get_feature_names_out()[col]
                if term == "dummy_text":
                    continue
                score = sum(tfidf_matrix[i, col] for i in range(tfidf_matrix.shape[0]))
                tfidf_scores[term] = score
            
            # Sort by score
            sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get top keywords
            top_keywords = [term for term, _ in sorted_terms[:top_n]]
            
            # If no keywords found, fall back to simple method
            if not top_keywords:
                return self.extract_keywords_simple(top_n)
                
            return top_keywords
            
        except Exception as e:
            print(f"TF-IDF extraction failed: {e}. Falling back to simple keyword extraction.")
            return self.extract_keywords_simple(top_n)
    
    def determine_conversation_nature(self, keywords):
        """
        Determine the nature of the conversation based on keywords.
        
        Args:
            keywords (list): List of extracted keywords
            
        Returns:
            str: Description of the conversation nature
        """
        # This is a simple implementation and could be made more sophisticated
        if not keywords:
            return "The conversation topic could not be determined."
        
        # Join keywords for a simple description
        return f"The conversation was mainly about {', '.join(keywords)}."
    
    def generate_summary(self, use_tfidf=True):
        """
        Generate a summary of the chat log.
        
        Args:
            use_tfidf (bool): Whether to use TF-IDF for keyword extraction
            
        Returns:
            str: Summary of the chat log
        """
        # Get message statistics
        stats = self.get_message_stats()
        
        # Check if we have any messages
        if stats['total_messages'] == 0:
            return "No messages found in the chat log."
        
        # Extract keywords
        try:
            if use_tfidf:
                keywords = self.extract_keywords_tfidf()
            else:
                keywords = self.extract_keywords_simple()
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            keywords = []
        
        # Handle cases where no keywords were found
        if not keywords:
            keywords_text = "No significant keywords found."
        else:
            keywords_text = f"Most common keywords: {', '.join(keywords)}."
        
        # Determine conversation nature
        conversation_nature = self.determine_conversation_nature(keywords)
        
        # Generate summary
        summary = f"""Summary:
- The conversation had {stats['exchanges']} exchanges ({stats['total_messages']} total messages).
- The user sent {stats['user_messages']} messages and the AI sent {stats['ai_messages']} messages.
- {conversation_nature}
- {keywords_text}
"""
        return summary

def process_single_file(file_path, use_tfidf=True):
    """
    Process a single chat log file.
    
    Args:
        file_path (str): Path to the chat log file
        use_tfidf (bool): Whether to use TF-IDF for keyword extraction
        
    Returns:
        str: Summary of the chat log
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    
    # Check if file is empty
    if os.path.getsize(file_path) == 0:
        return f"Error: File '{file_path}' is empty."
        
    summarizer = ChatLogSummarizer()
    
    if summarizer.parse_chat_log(file_path):
        return summarizer.generate_summary(use_tfidf)
    else:
        return "Failed to parse the chat log."

def process_directory(directory_path, use_tfidf=True):
    """
    Process all .txt files in a directory.
    
    Args:
        directory_path (str): Path to the directory containing chat logs
        use_tfidf (bool): Whether to use TF-IDF for keyword extraction
        
    Returns:
        dict: Dictionary mapping file names to summaries
    """
    results = {}
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return results
        
    # Check if it's actually a directory
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a directory.")
        return results
    
    # Get list of text files
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"No .txt files found in directory '{directory_path}'.")
        return results
    
    # Process each file
    for filename in txt_files:
        try:
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {filename}...")
            results[filename] = process_single_file(file_path, use_tfidf)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results[filename] = f"Failed to process: {str(e)}"
    
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