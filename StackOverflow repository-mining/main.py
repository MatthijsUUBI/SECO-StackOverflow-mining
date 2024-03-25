import numpy as np
import lda as lda_module
import dataretrieval
import preprocessing

# Path to the downloaded CSV file
csv_file_path = "QueryResults_21-24.csv"

# Retrieve data
df = dataretrieval.retrieve_data(csv_file_path)

# Preprocess data
preprocessed_data = preprocessing.preprocess(df)


# Convert text data to bag of words representation
vocab = list(set([word for doc in preprocessed_data for word in doc]))
doc_term_matrix = np.zeros((len(preprocessed_data), len(vocab)), dtype=int)
for i, doc in enumerate(preprocessed_data):
    for word in doc:
        doc_term_matrix[i, vocab.index(word)] += 1

# Create LDA model
model = lda_module.LDA(n_topics=10, n_iter=5000, random_state=1)
model.fit(doc_term_matrix)

# Code snippet for printing topics 1 to 10
n_top_words = 40

doc_topic = model.doc_topic_

print("\n")
for i in range(len(model.topic_word_)):  # Print topics 1 to 10
    topic_dist = model.topic_word_[i]
    topic_words = np.array(vocab)[np.argsort(topic_dist)][: -(n_top_words + 1) : -1]
    print("Topic {}: {}".format(i, " ".join(topic_words)))


def count_topic_prevalence(top_topics):
    topic_counts = {}  # Dictionary to store topic counts
    # Iterate through the top topics array and count occurrences of each topic
    for i in range(len(top_topics)):
        if top_topics[i].argmax() in topic_counts:
            topic_counts[top_topics[i].argmax()] += 1
        else:
            topic_counts[top_topics[i].argmax()] = 1

    # Calculate distribution percentages
    # total_topics = len(top_topics)
    topic_distribution = {topic: count for topic, count in topic_counts.items()}

    return topic_distribution


topic_distribution = count_topic_prevalence(doc_topic)
print("\n Topic Distribution:")
for topic, prevalence in topic_distribution.items():
    print(f"Topic {topic}: {prevalence}")
