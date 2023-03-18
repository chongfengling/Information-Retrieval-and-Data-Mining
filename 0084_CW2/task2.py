from gensim.models import Word2Vec
from BM25 import load_document, tokenisation, select_top_passages
import nltk
from tqdm import tqdm
import pandas as pd
import numpy as np


class LogisticRegression:
    def __init__(
        self,
        lr: float = 0.001,
        batch_size: int = 5000,
        tol: float = 0.0001,
        # max_iter: int = 10000,
        num_epochs: int = 100,
    ) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.tol = tol
        # self.max_iter = max_iter
        self.num_epochs = num_epochs
        self.weights = None
        self.train_loss_lst = []
        self.val_loss_lst = []

    def fit(self, X_train, y_train, X_val, y_val):
        m_train, n = X_train.shape
        # self.weights = np.ones(n) * 0.5
        self.weights = np.zeros(n)
        # self.weights = np.random.normal(0, 1, size=n)
        # initial loss is inf
        pre_loss = 10e10
        # epoch
        epoch = 0
        # early stop when there are no improvement of val loss in three continuing epochs
        no_improvement = 0
        while epoch < self.num_epochs:
            print(f'current epochs = {epoch}')
            # for each epoch, fits train data and computes loss
            index_shuffled = np.random.permutation(m_train)
            X_train_shuffled = X_train[index_shuffled, :]
            y_train_shuffled = y_train[index_shuffled]
            for i in range(0, m_train, self.batch_size):
                # if not divisible in the final batch, return all remaining
                X_train_batch = X_train_shuffled[i : min(i + self.batch_size, m_train)]
                y_train_batch = y_train_shuffled[i : min(i + self.batch_size, m_train)]
                # prediction for train data
                y_train_batch_pred = self.predict_prob(X_train_batch)
                # compute gradient
                grad = self._gradient(
                    y_true=y_train_batch,
                    y_pred=y_train_batch_pred,
                    X_train=X_train_batch,
                )
                # update the weights in every batch
                self.weights -= grad * self.lr
            # compute predictions and train/val loss in every epoch
            y_train_shuffled_pred = self.predict_prob(X_train_shuffled)
            y_val_pred = self.predict_prob(X_val)
            train_loss = self._compute_loss(
                y_true=y_train_shuffled, y_pred=y_train_shuffled_pred
            )
            val_loss = self._compute_loss(y_true=y_val, y_pred=y_val_pred)
            # record the training history
            self.train_loss_lst.append(train_loss)
            self.val_loss_lst.append(val_loss)
            # print info
            print(f'train loss = {train_loss}, validation loss = {val_loss}.')
            # early stop criterion
            if pre_loss - val_loss < self.tol:
                no_improvement += 1
            if no_improvement >= 4:
                break
            pre_loss = val_loss
            # next epoch
            epoch += 1

    def predict_prob(self, X: np.array) -> np.array:
        """return probability

        Parameters
        ----------
        X : np.array
            shape=(num_data, num_feature)

        Returns
        -------
        np.array
            shape=(num_data,)
        """
        return self._sigmoid(np.dot(X, self.weights))

    def predict(self, X: np.array, threshold: float = 0.5) -> list:
        """return estimated label (0 or 1)

        Parameters
        ----------
        X : np.array
            shape=(num_data, num_feature)
        threshold : float, optional
            _description_, by default 0.5

        Returns
        -------
        list
            len=num_data
        """
        y_pred = self.predict_prob(X)
        return list(map(int, y_pred >= threshold))

    def _sigmoid(self, x: np.array)->np.array:
        return 1 / (1 + np.exp(-x))

    def _gradient(self, y_true: np.array, y_pred: np.array, X_train: np.array)->np.array:
        """gradients of cross-entropy loss

        Parameters
        ----------
        y_true : np.array
            _description_
        y_pred : np.array
            _description_
        X_train : np.array
            _description_

        Returns
        -------
        np.array
            _description_
        """
        n = X_train.shape[0]
        return -(1 / n) * np.dot(np.transpose(X_train), (y_true - y_pred))

    def _compute_loss(self, y_true: np.array, y_pred: np.array) -> float:
        """cross-entropy loss

        Parameters
        ----------
        y_true : np.array
            true label
        y_pred : np.array
            predicted probability

        Returns
        -------
        float
            loss
        """
        n = len(y_true)
        return -(1 / n) * np.sum(
            np.multiply(y_true, np.log(y_pred))
            + np.multiply((1 - y_true), np.log(1 - y_pred))
        )

    def matric(self):
        pass


def text_preprocess(
    data: pd.DataFrame, save_name: str = None, do_subsample: bool = True
) -> pd.DataFrame:
    """from texts to vectors

    Parameters
    data : pd.DataFrame
        dataset
    save_name : str, optional
        save or not, by default None
    do_subsample: bool, optional
        subsample for train dataset only

    Returns
    -------
    pd.DataFrame
        the whole embedding dataset with columns []
    """
    # sample: subsample
    print('Estimated Time: about 3 mins on M1 Pro')
    if do_subsample:
        print("Start sampling ...")
        sampled_df = subsampling(data).reset_index(drop=False)
    else:
        sampled_df = data.reset_index(drop=False)
    # tokenisation of query and passage text
    print("Start tokenisation ...")
    tqdm.pandas(desc='Tokenisation(1/2) ...')
    sampled_query = sampled_df['queries'].progress_apply(tokenisation, remove=True)
    tqdm.pandas(desc='Tokenisation(2/2) ...')
    sampled_passage = sampled_df['passage'].progress_apply(tokenisation, remove=True)
    # build model
    print("Start building model ...")
    query_model = Word2Vec(
        sentences=sampled_query, vector_size=100, window=5, min_count=1, workers=4
    )
    passage_model = Word2Vec(
        sentences=sampled_passage, vector_size=100, window=5, min_count=1, workers=4
    )
    # average embedding
    print("Start embedding ...")
    query_ae = average_embedding(model=query_model, sentences=sampled_query)
    passage_ae = average_embedding(model=passage_model, sentences=sampled_passage)
    # create new pd.dataFrame
    res_df = pd.concat(
        [
            sampled_df['index'],
            sampled_df['qid'],
            sampled_df['pid'],
            pd.DataFrame(
                list(zip([1] * len(query_ae), query_ae, passage_ae)),
                columns=['intercept', 'query_ae', 'passage_ae'],
            ),
            sampled_df['relevancy'],
        ],
        axis=1,
    )
    if save_name:
        # difficult to load (str to array)
        # res_df.to_csv(f'input_df_{save_name}.csv')

        # save X, y into a .npy file
        X = np.asarray(
            res_df.apply(
                lambda x: np.hstack((x['intercept'], x['query_ae'], x['passage_ae'])),
                axis=1,
            ).values.tolist()
        )
        y = np.asarray(res_df['relevancy']).reshape(-1, 1)
        # X_y = np.hstack((X, y))
        # save qid, pid for later use
        qid, pid = np.asarray(res_df['qid']).reshape(-1, 1), np.asarray(
            res_df['pid']
        ).reshape(-1, 1)
        X_y = np.hstack((qid, pid, X, y))
        np.save(f'input_df_{save_name}.npy', X_y)

    return res_df


def subsampling(
    raw_df: pd.DataFrame = None,
    samples_n: int = 20,
    to_csv: bool = True,
    seed: int = None,
) -> pd.DataFrame:
    """sample from a big dataset. All positive samples remained, select samples_n negative samples for each label

    Parameters
    ----------
    raw_df : pd.DataFrame, optional
        big dataset, by default None
    samples_n : int, optional
        number of negative samples selected, by default 20
    to_csv : bool, optional
        if save the dataframe, by default True
    seed : int, optional
        random seed, by default None

    Returns
    -------
    pd.DataFrame
        the smaller dataset after sampling
    """
    if seed:
        np.random.seed(seed=seed)
    else:
        # works as  filename
        seed = 'None'
    # split positive and negative samples. sorted for iteration
    rel_df, irrel_df = raw_df[raw_df['relevancy'] == 1], raw_df[
        raw_df['relevancy'] == 0
    ].sort_values('qid')
    # get the number of negative samples for each label
    irrel_df_freq = irrel_df['qid'].value_counts()
    # iterates from the first row of raw_df
    index = 0
    # store the selected rows' index
    samples_index = []
    for i in irrel_df['qid'].unique():
        qid_num = irrel_df_freq[i]
        # no enough negative samples
        if qid_num <= samples_n:
            samples_index.extend(list(range(index, index + qid_num)))
        # selected samples randomly
        else:
            samples_index.extend(np.random.randint(index, index + qid_num, samples_n))
        # start at different index. raw_df sorted by qid
        index += qid_num
    # return selected negative samples
    irrel_sampled_df = irrel_df.iloc[samples_index]
    # concatenate two dataframe
    sampled_df = pd.concat([rel_df, irrel_sampled_df]).sort_index()
    # save or not
    if to_csv:
        sampled_df.to_csv(f'{seed}_sampled.csv')
    return sampled_df


def average_embedding(model, sentences):
    """average embedding for a list of tokens

    Parameters
    ----------
    model : Word2Vec model
        _description_
    sentences : list
        processed query/passage

    Returns
    -------
    list
        average embeddings of listx of tokens
    """
    res = []
    for i in tqdm(sentences, desc="Average Embedding ..."):
        # not empty
        if i:
            words_vectors = model.wv[i]
            res.append(np.mean(words_vectors, axis=0))
        # empty:
        else:
            res.append(np.zeros(model.vector_size))
    return res


if __name__ == '__main__':
    '''process and save to .npy files
    train_data = load_document(
        '/Users/ling/MyDocuments/COMP0084/0084_CW2/train_data.tsv', names=None
    )
    text_preprocess(train_data, save_name='train')

    val_data = load_document(
        '/Users/ling/MyDocuments/COMP0084/0084_CW2/validation_data.tsv', names=None
    )
    text_preprocess(val_data, save_name='val', do_subsample=False)
    '''

    test_data = load_document(
        '/Users/ling/MyDocuments/COMP0084/0084_CW2/candidate_passages_top1000.tsv', names=['qid', 'pid', 'queries', 'passage']
    )
    test_data['relevancy'] = [-1] * len(test_data['qid'])
    text_preprocess(test_data, save_name='test', do_subsample=False)
    # '''

    # load from .npy file
    # intercept 1 + embedding 200 + label 1
    # shape = (95874, 202)
    Xy_train = np.load('input_df_train_sub.npy')
    # shape = (1103039, 202)
    Xy_val = np.load('input_df_val_nosub.npy')

    X_train, y_train = Xy_train[:, :-1], Xy_train[:, -1]
    X_val, y_val = Xy_val[:, :-1], Xy_val[:, -1]

    LR_model = LogisticRegression()
    LR_model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    # LR_model.predict_prob(X_val)
    pass
