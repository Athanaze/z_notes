LDA (Latent Dirichlet Allocation)

Example of documents  : every doc is a mix of diff topics

Every topic is characterized by a distribution of words. The "Politics" topic is likely to contain words like "election," "vote," "candidate," "party," "government" frequently. The "Economics" topic might have "market," "stock," "trade," "inflation," "GDP."

LDA is a generative model : it assumes a hypothetical process for how the documents you see could have been generated. 

So we have to image we want to write a new document

We pick the proportions of topics : the mix is influenced by the Dirichlet distribution
For each word in the document we first pick  which topic it will cover. The probability of each topic is based on the mix we choose at step 1.
We then pick a word based on the topic we choose. This time the probability of the word we choose is based on the distribution of words for that topic
Repeat that process for each word we want to have in the document

LDA doesn't actually generate documents. It observes the finished documents (just the sequence of words) and tries to work backward to figure out the hidden structure:
Most probable mix of topics
Most probable word distribution for each topic
Which topic likely generated each specific word occurrence in each document?$


Initialization

The algorithm starts with a random (or semi-random) assignment of topics to each word instance in the corpus. The initial topics are meaningless garbage.

Then iteration by iteration, the algorithm looks at one word at a time and for each of them decide which topic should have generated it

-> how well does this word fit with each topic’s current word distribution:
 Based on al the others words currently assigned to Topic X across the entire corpus, how likely is “gene” to belong to topic X ?
- > Based on all the other words in this specific document, what’s the current estimate of its topic mixture

Using the response from those 2 questions, we re assign the word to a topic
-> if “GENE” frequently appears in documents that also contain “DNA” and “sequence”, words assigned to topic 3, then “gene” becomes more likely to be assigned to topic 3.
We consider both GLOBAL TOPIC DEFINITION and LOCAL DOCUMENT CONTEXT

The algorithm converges when the topic assignment stabilize

UNSUPERVISED


-> hyperparameter K, the number of topics we want to extract

-> The topics emerge purely from the statistical patterns of word co-occurrence in the documents.



# Projection matrix : self-adjointness property

![2025-03-10-095524_650x43_scrot](https://github.com/user-attachments/assets/7a435f4b-63eb-46e5-b530-1904ce8d6892)


![2025-03-10-094909_663x131_scrot](https://github.com/user-attachments/assets/917d5d90-c090-40b4-8626-4c5d1f8bde3f)


# CGAL

```
K::Point_2 a;
K::Point_2 b;

CGAL::right_turn(a, b, c) // true if c makes a right turn relative to the directed line segment from a to b 
```

# AML

# Conditional entropy

$H(X \mid Y) = - \sum_{y \in \mathcal{Y}} \sum_{x \in \mathcal{X}} P(Y = y) P(X = x \mid Y = y) \log P(X = x \mid Y = y)$

$= - \sum_{x \in \mathcal{X}, y \in \mathcal{Y}} P(X = x, Y = y) \log \frac{P(X = x, Y = y)}{P(Y = y)},$

# Mutual information

$I(X; Y) := H(X) - H(X \mid Y)$

Measures the amount of information of X left after Y is revealed.

![2025-01-03-184515_818x62_scrot](https://github.com/user-attachments/assets/88b623b4-0a0e-4ad9-a714-ddecd518e070)

# Matrix calculus

We have

![2025-01-03-180910_189x110_scrot](https://github.com/user-attachments/assets/289a2781-670c-48c2-8fff-409102f5cc00)

# Recursion pro tip : if there are some overlap, just substract it !

![2024-12-16-151305_1733x397_scrot](https://github.com/user-attachments/assets/8532118d-b2fc-452b-bc04-ca0a15491991)

