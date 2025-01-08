# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:34:55 2024

@author: xristos
"""

#important libraries for requesting and scraping wiki pages
import requests
from bs4 import BeautifulSoup
import json
from collections import defaultdict
import time
#important libraries for text tokkenization
import nltk 
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
import math

#COLLECT WIKI PAGES

#create a wikipedia crawler using beautifulsoup
#to parse html and extract the text 

# List of wikipedia urls to scrape
#every url we save here is going to be fetched
start_urls = [
    "https://en.wikipedia.org/wiki/Dot-com_bubble",
    "https://en.wikipedia.org/wiki/Real_estate_investment_trust",
    "https://en.wikipedia.org/wiki/2007%E2%80%932008_financial_crisis",
    "https://en.wikipedia.org/wiki/COVID-19_recession",
    "https://en.wikipedia.org/wiki/Financial_crisis",
    "https://en.wikipedia.org/wiki/S%26P_500",
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "https://en.wikipedia.org/wiki/Nasdaq",
    "https://en.wikipedia.org/wiki/Cryptocurrency",
    "https://en.wikipedia.org/wiki/Energy_industry",
    "https://en.wikipedia.org/wiki/Goldman_Sachs",
    "https://en.wikipedia.org/wiki/Stock_market_crash",
    "https://en.wikipedia.org/wiki/Category:Real_estate_companies_of_the_United_States",
    "https://en.wikipedia.org/wiki/Blockchain",
    "https://en.wikipedia.org/wiki/Bitcoin"
    
]

#FETCH_ARTICLE FUNCTION
# Function to extract the content of a Wikipedia article
def fetch_article(url):
    #send a request and receive a status code (200means succesfull request)
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve {url}")
        return None
    
    #extract html parts
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('h1').text                                #extract header
    content = ' '.join([p.text for p in soup.find_all('p')])    #extract paragraphs
    
    #in some urls I gave (sp500companies) there are tables with crucial info so lets check for it too
    tables = []
    for table in soup.find_all('table', {"class": "wikitable"}):
        headers = [header.text.strip() for header in table.find_all('th')]
        rows = []
        
        for row in table.find_all('tr'):
            cells = [cell.text.strip() for cell in row.find_all('td')]
            if cells:
                # Create a dictionary for each row with headers as keys
                row_data = dict(zip(headers, cells))
                rows.append(row_data)
        
        # Append table with headers and rows to tables list
        tables.append({
            "headers": headers,
            "rows": rows
        })
    
    return {
        "title": title,
        "content": content,
        "url": url,
        "tables": tables
    }

#SCRAPE_WIKIPEDIA_ARTICLES FUNCTION
#function to scrape multiple articles and save them to a JSON file
def scrape_wikipedia_articles(urls, output_file="wikipedia_articles.json"):
    articles = []
    for url in urls:
        print(f"Scraping {url}")
        article = fetch_article(url)
        if article:
            articles.append(article)
        #Give one second delay in order not to lag the server
        time.sleep(1)  

    #save the articles to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(articles)} articles to {output_file}")


#PREPROCESSING STAGE

#create a list with all stopwords in english language
stop_words =  set(stopwords.words('english'))
#lemmatize each word to the original
lemmatizer = WordNetLemmatizer()

#Preprocess words
#function to substract each word from the text and hold the originals
def preprocess_text(text):
    #remove everything in brackets that wikipedia has
    text = re.sub(r'\[[^\]]*\]', '', text)  # Remove anything in square brackets
    #tokenize text to lower case to avoid duplicates
    tokens = word_tokenize(text.lower())
    #remove stopwords and lemmatize 
    originals = [
        lemmatizer.lemmatize(token) for token in tokens
        if token.isalpha() and token not in stop_words
        ]
    #return main root of every word
    return ' '.join(originals)
    

#Preprocess articles
#function to preprocess articles
def preprocess_articles(input_file="wikipedia_articles", output_file="original_articles.json"):
    with open(input_file,"r",encoding="utf-8") as f:
        articles = json.load(f)
        
        #read each article
        for article in articles:
            print(f"Processing article: {article['title']}")
            article['original_words'] = preprocess_text(article['content'])
            #process tables if they exist
            if "tables" in article:
                for table in article["tables"]:
                    for row in table["rows"]:
                        #take the word of each cell
                        for key in row:
                            row[key]=preprocess_text(row[key])
        
        #save the result that occurs after process
        with open (output_file,"w",encoding='utf-8') as f:
            json.dump(articles,f,ensure_ascii=False,indent=4)
        print(f"Saved preprocessed articles to {output_file}")


#INDEXING STAGE
#create a list for every token with indexes. Every index represents the article the token is in
def build_inverted_index(input_file="original_articles.json", output_file="inverted_index.json"):
    with open(input_file,'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    #create an index for every token that works as a list
    #then append to it the article id that the token is shown    
    inverted_index = defaultdict(list) 
    
    #hold an id of every article
    for doc_id, article in enumerate(articles):
        print (f"indexing article: {article['title']}")
        #tokenize everything between spaces
        terms = article['content'].split()
        
        #append the article id in the list of every token as set above
        for term in set(terms):
            inverted_index[term].append(doc_id)
            
        #save inverted index
        with open(output_file,'w', encoding='utf-8') as f:
            json.dump(inverted_index,f,ensure_ascii=False,indent=4)
        print(f"saved inverted index to {output_file}")
        
        
#EXAMINE INVERTED INDEX
#check some words to see the total of them and how many articles include them
def test_index(input_file="inverted_index.json", term_to_check=""):
    with open(input_file, 'r', encoding='utf-8') as f:
        inverted_index = json.load(f)

    #print hoy many times the word is appeared in all articles
    print(f"Inverted index loaded with {len(inverted_index)} terms.")
    
    #print the exact articles the term is in
    #based in inverted index we have made
    if term_to_check:
        documents = inverted_index.get(term_to_check, [])
        print(f"Term '{term_to_check}' found in documents: {documents}")
    else:
        # Print the first few terms for review
        print("Sample of terms in the inverted index:")
        for i, term in enumerate(inverted_index):
            print(f"  {term}: {inverted_index[term]}")
            if i >= 4:  # Limit to 5 terms for clarity
                break


#QUERY PROCESSING
#this part include tokenization of query and search inside articles
#logical operations between query tokens will be calculated to return relevant articles
def process_query(query,index_file="inverted_index.json",articles_file="original_articles.json"):
    #load the inverted index with all info
    with open(index_file,'r',encoding='utf-8') as f:
        inverted_index=json.load(f)
    
    #take a query tokkenize and split it by spaces
    query = query.lower()
    tokens = query.split()
    
    #usually programmers or some experienced searchers use operators for better results
    #so declare an operator
    result=set()
    current_operator= None
    
    for token in tokens:
        if token == 'and':
            current_operator="AND"
        elif token =='or':
            current_operator="OR"
        elif token == 'not':
            current_operator="NOT"
        else:
            #take document ids that include the term
            docs = set(inverted_index.get(token,[]))
            #calculate boolean operations
            if current_operator=="OR":
                result |= docs 
            elif current_operator=="NOT":
                result -= docs #exclude documents
            elif current_operator =="AND" or current_operator is None: #AND is the default operator
                result = result & docs if result else docs #if the set is empty make docs the result

    #return the titles and IDs of the token
    with open(articles_file,'r',encoding='utf-8') as f:
        articles = json.load(f)
    
    matching_docs = [{"id": doc_id, "title":articles[doc_id]['title']} for doc_id in result]
    return matching_docs



#RANKING
#this step aims to sort every article based on the relevance it has with the given query
##TF-IDF algorithm will help succeed thiss 

def calculate_tf(articles,output_file="tf.json"):
    #declare a list that shows the frequency of a word in a specific article 
    tf = {}
    for doc_id,article in enumerate(articles):
        tokens = article['content'].split()
        term_counter = defaultdict(int)
        
        for term in tokens:
            term_counter[term] +=1
        
        #make counter as frequency
        tf[doc_id] = {term: count / len(tokens) for term, count in term_counter.items()}
        
        #save to a file to undestand better
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tf, f, ensure_ascii=False, indent=4)
        
    return tf


#calculate idf for all terms
def calculate_idf(articles,tf,output_file="idf.json"):
    #total number of documents
    N = len(articles)
    df = defaultdict(int)
    
    #count how many documents include each term
    for doc_tf in tf.values():
        for term in doc_tf:
            df[term] +=1
    
    #compute IDF adding 1 to avoid collision with zero
    idf = {term: math.log(N / (1 + freq)) for term, freq in df.items()}
    
    #save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(idf, f, ensure_ascii=False, indent=4)
        
    return idf


#document ranking using tf-idf
def rank_docs_tf_idf(query,tf,idf,inverted_index,articles):
    query_attributes = query.lower().split()
    doc_scores = defaultdict(float)
    
    #calculate tf-idf score for each document
    for term in query_attributes:
        if term in idf:
            idf_term = idf[term]
            for doc_id in inverted_index.get(term,[]):
                tf_term=tf[doc_id].get(term,0)
                doc_scores[doc_id] += tf_term * idf_term
    
    ranked_docs = sorted(doc_scores.items(), key = lambda x: x[1],reverse=True)
    return [{"id": doc_id, "title": articles[doc_id]['title'], "score": score} for doc_id, score in ranked_docs]


#document ranking using boolean retreival
def rank_docs_boolean(query,inverted_index,articles):
    query_attributes = query.lower().split()
    result_set = set()

    #retrieve all documents that occur from operators and/or between tokens
    for term in query_attributes:
        if term in inverted_index:
            result_set.update(inverted_index[term])

    # Return all matching documents
    return [{"id": doc_id, "title": articles[doc_id]['title']} for doc_id in result_set]


#function to combine tf-idf and boolean choice
def rank_docs(query, tf, idf, inverted_index, articles, method):
    if method == "TF-IDF":
        return rank_docs_tf_idf(query, tf, idf, inverted_index, articles)
    elif method == "Boolean":
        return rank_docs_boolean(query,inverted_index,articles)
    else:
        raise ValueError("Unsupported ranking method. Choose 'TF-IDF' or 'Boolean'.")



#rank query results calculating tf and idf
def rank_query(query,method, index_file="inverted_index.json", articles_file="original_articles.json"):
    with open(index_file, 'r', encoding='utf-8') as f:
        inverted_index = json.load(f)
    with open(articles_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    #calculate tf
    tf = calculate_tf(articles)
    #calculate idf
    idf = calculate_idf(articles,tf)
    #rank docs
    ranked_results = rank_docs(query,tf,idf,inverted_index,articles,method=method)
    return ranked_results


#EVALUATING SYSTEM
def evaluate_engine(queries,relevance_list,articles_file="original_articles.json"):
    #load articles
    with open(articles_file,'r',encoding='utf-8') as f:
        articles = json.load(f)
        
    #store results
    precision_scores = []
    recall_scores = []
    f1_scores = []
    average_precisions = []

    for query, relevant_docs in relevance_list.items():
        
        results = rank_query(query,method="TF-IDF")
        retrieved_docs = [result['id'] for result in results]
        
        # Calculate relevant retrieved documents
        relevant_retrieved = set(retrieved_docs) & set(relevant_docs)
       
        # Precision
        precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
        precision_scores.append(precision)
       
        # Recall
        recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
        recall_scores.append(recall)

        # F1-Score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        f1_scores.append(f1)

        # Average Precision for MAP
        ap = 0
        correct_hits = 0
        for i, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in relevant_docs:
                correct_hits += 1
                ap += correct_hits / i
        ap /= len(relevant_docs) if relevant_docs else 1
        average_precisions.append(ap)
    
    # Aggregate metrics
    mean_precision = sum(precision_scores) / len(precision_scores)
    mean_recall = sum(recall_scores) / len(recall_scores)
    mean_f1 = sum(f1_scores) / len(f1_scores)
    map_score = sum(average_precisions) / len(average_precisions)
    
    return {
        "Precision": mean_precision,
        "Recall": mean_recall,
        "F1-Score": mean_f1,
        "MAP": map_score
    }



#run article crawler
scrape_wikipedia_articles(start_urls)
#preprocess articles
preprocess_articles("wikipedia_articles.json")
#build index that shows where every word is included
build_inverted_index()

#test index results synopsis
#test_index()

#run a query and print results with tf-idf method
query = "what is bitcoin" 
results = rank_query(query,method="TF-IDF")
print(f"Ranked results for query '{query}':")
for result in results:
    print(f"Document ID: {result['id']}, Title: {result['title']}, Score: {result['score']:.4f}")

#for the same query return results with boolean method
results = rank_query(query,method="Boolean")
print(f"Ranked results for query '{query}':")
for result in results:
    print(f"Document ID: {result['id']}, Title: {result['title']}")


    
#evaluate engine
query_relevance = {
    "Financial Crisis": [0, 2, 3, 4, 11],
    "Covid Economy Consequences": [3],
    "What companies Are included to nasdaq": [5,6,7,9],
    "Banks of America": [6,7,10],
    "Real Estate Investments": [1,12],
    "Blockchain and Crypto": [8,13,14],
}

metrics = evaluate_engine(query_relevance.keys(), query_relevance)
print("Evaluation Results:")
print(metrics)
