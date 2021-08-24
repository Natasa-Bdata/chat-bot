#Import the libraries used for counting the tf
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm

#import dataset using the pandas
dataset = pd.read_csv('insurance_qna_dataset.csv', sep='\t').iloc[:, 1:]
dataset_group_by_question = dataset.groupby('Question', as_index=False).agg(lambda x: np.unique(x).tolist())
#dataset_delete_nanValues = dataset.dropna(axis=0)
#dataset_groupBy_question = dataset_delete_nanValues.groupby('Question', as_index=False).agg({'Answer': lambda d: ', '.join(set(d))})
dataset_question = dataset_group_by_question.iloc[:, 0]

#Using the TdVectorizer to calculate the tf value for words in questions
tf = CountVectorizer()
count_tf = tf.fit_transform(dataset_question)

#Ask an insurance related question
new_question = [input("Please enter your insurance related question: ")]
count_tf_new_question = tf.transform(new_question)
print('\n')

number_of_question = len(dataset_question)
count_tf_new_question_todense = count_tf_new_question.toarray()
count_tf_todense = count_tf.toarray()
number_of_words = len(count_tf_todense[0])

'''
print("This are the most similar questions according to manhattan distances: ")
new_list = []
for question in range(0, number_of_question):
    num_q = question
    calculate_md = 0
    test = [num_q]
    for values in range(0, number_of_words-1):
        calculate_md = calculate_md + np.absolute(count_tf_todense[num_q, values] - count_tf_new_question_todense[0, values])
    test.append(calculate_md)
    new_list.append(test)
new_list = sorted(new_list, key=lambda x: x[1])
manhatt_dist = np.array(new_list)

for result in range(0, 5):
    best_match = new_list[result][0]
    print(dataset_question[best_match])
'''
the_top_similar = 10
print(f"This are the most {the_top_similar} similar questions according to manhattan distances: \n")

new_list = []
for question in range(0, number_of_question):
    num_q = question
    calculate_md = 0
    test = [num_q]
    calculate_md = np.sum(np.absolute(count_tf_todense[num_q, :] - count_tf_new_question_todense[0, :]))
    test.append(calculate_md)
    new_list.append(test)
new_list = sorted(new_list, key=lambda x: x[1])
manhatt_dist = np.array(new_list)

for result in range(0, the_top_similar):
    best_match = new_list[result][0]
    print(dataset_question[best_match])
print('\n\n')

print(f"This are the most {the_top_similar} similar question according to euclidean distances: \n")
new_list = []
for question in range(0, number_of_question):
    num_q = question
    calculate_ed = 0
    test = [num_q]
    calculate_ed = np.sqrt(np.sum((count_tf_todense[num_q, :] - count_tf_new_question_todense[0, :])**2))
    test.append(calculate_ed)
    new_list.append(test)
new_list = sorted(new_list, key=lambda x: x[1])
euclid_dist = np.array(new_list)

for result in range(0, the_top_similar):
    best_match = new_list[result][0]
    print(dataset_question[best_match])
print('\n\n')


print(f"This are the most {the_top_similar} similar question according to cosine similarity: \n")
new_list = []
for question in range(0, number_of_question):
    num_q = question
    calculate_cs = 0
    test = [num_q]
    #calculate_cs = np.sum(count_tf_todense[num_q, :] * count_tf_new_question_todense[0, :])/(np.sum(count_tf_todense[num_q, :]**2)) * np.sum(count_tf_new_question_todense[0, :]**2)
    calculate_cs = dot(count_tf_todense[num_q, :], count_tf_new_question_todense[0, :]) / (norm(count_tf_todense[num_q, :]) * norm(count_tf_new_question_todense[0, :]))
    test.append(calculate_cs)
    new_list.append(test)
new_list = sorted(new_list, key=lambda x: x[1])
cosine_dist = np.array(new_list)

for result in range(1, the_top_similar+1):
    best_match = new_list[-result][0]
    print(dataset_question[best_match])
print('\n\n')

'''
print("This are the most similar question according to euclidean distances: ")
new_list = []
for question in range(0, number_of_question):
    num_q = question
    calculate_ed = 0
    test = [num_q]
    for values in range(0, number_of_words-1):
        calculate_ed = calculate_ed + (count_tf_todense[num_q, values] - count_tf_new_question_todense[0, values])**2
    test.append(np.sqrt(calculate_ed))
    new_list.append(test)
new_list = sorted(new_list, key=lambda x: x[1])
euclid_dist = np.array(new_list)

for result in range(0, 5):
    best_match = new_list[result][0]
    print(dataset_question[best_match])
'''

manhattan_dist = manhattan_distances(count_tf.toarray(), count_tf_new_question.toarray()).flatten()
print("This are the most similar questions according to manhattan distances: ")
print(dataset_question[manhattan_dist.argsort() [:the_top_similar]])


euclidean_dist = euclidean_distances(count_tf.toarray(), count_tf_new_question.toarray()).flatten()
print("This are the most similar questions according to euclidean distances: ")
print(dataset_question[euclidean_dist.argsort() [:the_top_similar]])

cosine_similarity = cosine_similarity(count_tf.toarray(), count_tf_new_question.toarray()).flatten()
print("This are the most similar questions according to cosine similarity: ")
print(dataset_question.iloc[cosine_similarity.argsort() [-the_top_similar:][::-1]])







