H παρούσα μηχανή αναζήτησης αξιοποιεί δύο μεθόδους κατάταξης εγγράφων

1.TF-IDF (Term Frequency-Inverse Document Frequency): Χρησιμοποιείται για την κατάταξη εγγράφων με βάση τη συχνότητα όρων και τη μοναδικότητά τους.
2. Boolean Retrieval: Χρησιμοποιείται για την αναζήτηση εγγράφων που περιέχουν συγκεκριμένους όρους χωρίς βαθμολογίες.
Η μηχανή υποστηρίζει δημιουργία αρχείων JSON, τα οποία ήδη περιέχουν:

Κάθε αρχείο περιέχει:
search_engine.py : Πηγαίος κώδικας μηχανής αναζήτησης
inverted_index.json : Το αντίστροφο ευρετήριο (inverted index) για τα άρθρα.
wikipedia_articles.json : Τα αρχικά άρθρα με τίτλους και περιεχόμενα.
wiki_page_urls : Τα url των wiki ιστοσελίδων που χρησομοποιήθηκαν.
original_articles.json: Τα αρχικά άρθρα με 'καθαρισμένο' περιεχόμενο.
TF.json : Συχνότητα ενός όρου σε κάθε άρθρο.
IDF.json : Συχνότητα όρου σε όλα τα έγγραφα.

Οδηγίες Χρήσης :
1. Αποθηκεύστε όλα τα αρχεία JSON που περιέχουν το inverted_index και τα άρθρα.
2. Τρέξτε τη συνάρτηση rank_query() με το query της επιλογής σας. Κατά προτίμηση χρησιμοποιείστε οικονομικούς όρους.
3. Eπιλέξτε μέθοδο αναζήτησης: "TF-IDF" ή "Boolean".

Παράδειγμα : 
query = "Covid Economy Consequences" 
results = rank_query(query,method="TF-IDF")

Εμφάνιση Αποτελεσμάτων:
print(f"Ranked results for query '{query}':")
for result in results:
    print(f"Document ID: {result['id']}, Title: {result['title']}")





