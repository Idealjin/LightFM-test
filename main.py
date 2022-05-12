import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
from scipy.io import mmwrite

# Data Load
# ratings_source: build_interactions 재료, list of tuples
# --> [(user1, item1), (user2, item5), ... ]
# item_features_source: build_item_features 재료
# --> [(item1, [feature, feature, ...]), (item2, [feature, feature, ...])]
ratings = pd.read_csv('data/ratings.csv')
ratings_source = [(ratings['user_id'][i], ratings['book_id'][i]) for i in range(ratings.shape[0])]

item_meta = pd.read_csv('data/books.csv')
item_meta = item_meta[['book_id', 'authors', 'average_rating', 'original_title']]

item_features_source = [(item_meta['book_id'][i],
                         [item_meta['authors'][i],
                          item_meta['average_rating'][i]]) for i in range(item_meta.shape[0])]

dataset = Dataset()
dataset.fit(users=ratings['user_id'].unique(),
            items=ratings['book_id'].unique(),
            item_features=item_meta[item_meta.columns[1:]].values.flatten()
            )

interactions, weights = dataset.build_interactions(ratings_source)
item_features = dataset.build_item_features(item_features_source)

# Save
mmwrite('data/interactions.mtx', interactions)
mmwrite('data/item_features.mtx', item_features)
mmwrite('data/weights.mtx', weights)

# Split Train, Test data
train, test = random_train_test_split(interactions, test_percentage=0.1)
train, test = train.tocsr().tocoo(), test.tocsr().tocoo()
train_weights = train.multiply(weights).tocoo()

from hyperopt import fmin, hp, tpe, Trials

#
# Define Search Space
trials = Trials()
space = [hp.choice('no_components', range(10, 50, 10)),
         hp.uniform('learning_rate', 0.01, 0.05)]

# Define Objective Function
def objective(params):
    no_components, learning_rate = params

    model = LightFM(no_components=no_components,
                    learning_schedule='adagrad',
                    loss='warp',
                    learning_rate=learning_rate,
                    random_state=0)

    model.fit(interactions=train,
              item_features=item_features,
              sample_weight=train_weights,
              epochs=3,
              verbose=False)

    test_precision = precision_at_k(model, test, k=5, item_features=item_features).mean()
    print("no_comp: {}, lrn_rate: {:.5f}, precision: {:.5f}".format(
      no_components, learning_rate, test_precision))
    # test_auc = auc_score(model, test, item_features=item_features).mean()
    output = -test_precision

    if np.abs(output+1) < 0.01 or output < -1.0:
        output = 0.0

    return output

best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

# Find Similar Items
model = LightFM(learning_schedule='adagrad',
                loss='warp',
                random_state=0)

model.fit(interactions=train,
          item_features=item_features,
          sample_weight=train_weights,
          epochs=3,
          verbose=False)

item_biases, item_embeddings = model.get_item_representations(features=item_features)

def make_best_items_report(item_embeddings, book_id, num_search_items):
    item_id = book_id - 1

    # Cosine similarity
    scores = item_embeddings.dot(item_embeddings[item_id])  # (10000, )
    item_norms = np.linalg.norm(item_embeddings, axis=1)  # (10000, )
    item_norms[item_norms == 0] = 1e-10
    scores /= item_norms

    # best: score가 제일 높은 item의 id를 num_search_items 개 만큼 가져온다.
    best = np.argpartition(scores, -num_search_items)[-num_search_items:]
    similar_item_id_and_scores = sorted(zip(best, scores[best] / item_norms[item_id]),
                                        key=lambda x: -x[1])

    # Report를 작성할 pandas dataframe
    best_items = pd.DataFrame(columns=['book_id', 'title', 'author', 'score'])

    for similar_item_id, score in similar_item_id_and_scores:
        book_id = similar_item_id + 1
        title = item_meta[item_meta['book_id'] == book_id].values[0][3]
        author = item_meta[item_meta['book_id'] == book_id].values[0][1]
        row = pd.Series([book_id, title, author, score], index=['book_id', 'title', 'author', 'score'])
        best_items = pd.concat([row])

        print(best_items)

    return best_items


# book_id 2: Harry Potter and the Philosopher's Stone by J.K. Rowling, Mary GrandPré
# book_id 9: Angels & Demons by Dan Brown
report01 = make_best_items_report(item_embeddings, 2, 10)
print("------------------------------------------------")
report02 = make_best_items_report(item_embeddings, 9, 10)