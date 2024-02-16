import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import re
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def get_wordnet_pos(word):
    """Map POS tag to a format recognized by lemmatize() function."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def preprocess_text(text):
    """Preprocess a given text by converting to lowercase, removing punctuation and stopwords."""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens if
                         w not in stopwords.words('english')]
    return lemmatized_tokens


# Article URLs and names
articles = ["Medieval Ship", "AI Fitness Trainers"]
texts = []

# Open the file "ship-timber-date.txt" for reading
with open("ship-timber-date.txt", "r") as ship_text:
    # Read the entire contents of the file into a variable
    ship_contents = ship_text.read()

# Open the file "ai-trainers.txt" for reading
with open("ai-trainers.txt", "r") as ai_text:
    # Read the entire contents of the file into a variable
    ai_contents = ai_text.read()

tokens_1 = preprocess_text(ship_contents)
tokens_2 = preprocess_text(ai_contents)
texts.append(tokens_1)
texts.append(tokens_2)

# Prepare Document-Term Matrix
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Apply LDA with increased passes and adjusted number of topics
lda_model = models.LdaModel(corpus, num_topics=len(articles), id2word=dictionary, passes=20)

# Task 1: Visualise the top 10 most important words for each topic
num_topics = len(articles)
fig, axes = plt.subplots(1, num_topics, figsize=(15, 10))  # Enlarged figure size for better readability

# If there is only one topic, matplotlib does not return axes as an array, so we wrap it in a list.
if num_topics == 1:
    axes = [axes]

# Plot the bar graphs for each topic
for i in range(num_topics):
    # Extract the top 10 words for the topic
    top_words = lda_model.show_topic(i, 10)
    # Convert top words into a DataFrame
    df = pd.DataFrame(top_words, columns=['term', 'prob']).set_index('term')
    # Plot a horizontal bar graph
    df.plot(kind='barh', ax=(axes[i] if num_topics > 1 else axes), title=f"Topic {i}", legend=False)
    axes[i].invert_yaxis()  # Display the highest probability term at the top

# Adjust the layout to ensure no overlaps and labels/titles are clear
plt.tight_layout()
plt.show()

# Task 2: Display and interpret the top 3 topics
top_topics = lda_model.top_topics(corpus, topn=3)
for idx, topic in enumerate(top_topics):
    print(f"Top {idx + 1} Topic Words and Weights:")
    print(topic[0])  # Each topic's word distribution
    print("\n")

# Task 3: Summarize the articles based on LDA topics
article_summaries = {}
for idx, article_title in enumerate(articles):
    bow = dictionary.doc2bow(texts[idx])
    topic_distribution = lda_model.get_document_topics(bow)
    dominant_topic = sorted(topic_distribution, key=lambda x: x[1], reverse=True)[0][0]
    topic_terms = lda_model.show_topic(dominant_topic, 5)
    topic_terms = [term for term, prob in topic_terms]
    summary = f"This article likely discusses {' '.join(topic_terms)}."
    article_summaries[article_title] = summary

for article_title, summary in article_summaries.items():
    print(f"{article_title}: {summary}\n")
