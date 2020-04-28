import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train(x, k, seed=1234):

    np.random.seed(seed)

    # Loading data
    ratings_path = 'movielens_1m/ratings.dat'
    ratings = pd.read_csv(ratings_path, sep="::", names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

    # Split into train/val sets
    train, test = train_test_split(ratings, train_size=x)

    # UserID, MovieID are not continuous, we need to create a dictionary whose keys are UserID/MovieID,
    # values are the corresponding actual indices in customer-product ratings matrix (train).
    unique_users = sorted(list(set(train.UserID)))
    unique_movies = sorted(list(set(train.MovieID)))
    user_map, movie_map = {}, {}
    for idx, userID in enumerate(unique_users):
        user_map[userID] = idx
    for idx, itemID in enumerate(unique_movies):
        movie_map[itemID] = idx
    n_users = len(unique_users)
    n_movies = len(unique_movies)

    # Convert the data above to a sparse m x n matrix, where each row is one user,
    # each column is one movie, and the value is the rating. This is the so-called
    # "customer-product ratings matrix".
    R = train.pivot(index='UserID', columns='MovieID', values='Rating')

    # Filling matrix using average ratings for a product
    for idx_movie in unique_movies:
        col = R[idx_movie]
        col.fillna(col.mean(), inplace=True)

    # Normalizing the matrix by subtraction of customer average for a product.
    R_norm = R.copy().to_numpy()
    mean_vals = []
    for idx_user in range(n_users):
        row = R_norm[idx_user]
        mean_val = np.mean(row)
        mean_vals.append(mean_val)
        R_norm[idx_user] -= mean_val
    mean_vals = np.array(mean_vals)

    # Compute the sparse SVD
    Uk, Sk, Vk_t = svds(R_norm, k=k)
    Sk_sqrt = np.sqrt(Sk)
    Sk_mat = np.diag(Sk_sqrt)
    U = np.matmul(Uk, Sk_mat)
    Vt = np.matmul(Sk_mat, Vk_t)
    V = np.transpose(Vt)

    # Defining the predictor
    def predict(U, V, mean_vals, idx_user, idx_item):
        mean_val = mean_vals[idx_user]
        user_vec = U[idx_user, :]
        item_vec = V[idx_item, :]
        pred_val = mean_val + np.inner(user_vec, item_vec)
        return pred_val

    # Computing MAE on test set
    test.reset_index(inplace=True)
    n_test_ratings = test.shape[0]
    true_vals = []
    pred_vals = []
    count = 0
    for idx in range(n_test_ratings):
        uID = test['UserID'][idx]
        mID = test['MovieID'][idx]
        if uID not in user_map or mID not in movie_map:
            continue
        idx_user = user_map[uID]
        idx_movie = movie_map[mID]
        true_val = test['Rating'][idx]
        pred_val = predict(U, V, mean_vals, idx_user, idx_movie)
        true_vals.append(true_val)
        pred_vals.append(pred_val)
        count += 1
    mae = mean_absolute_error(true_vals, pred_vals)
    print(f'\ntested on {count} ratings')

    return mae


if __name__ == '__main__':
    x_list = [0.2, 0.5, 0.8]
    k_list = [2, 5, 8, 11, 14, 25, 50, 100]
    errors = np.zeros((len(x_list), len(k_list)))
    plt.clf()
    plt.rcParams["savefig.dpi"] = 200
    plt.figure(figsize=[14, 6])
    markers = ['.', 's', 'o']
    for idx_x, x in enumerate(x_list):
        for idx_k, k in enumerate(k_list):
            mae = train(x, k)
            errors[idx_x, idx_k] = mae
        plt.plot(k_list, errors[idx_x, :], marker=markers[idx_x], label=f'x={x}')
    plt.legend()
    plt.xlabel('number of dimension, k')
    plt.ylabel('MAE')
    plt.xticks(k_list)
    plt.savefig('svd_prediction_quality.png')
