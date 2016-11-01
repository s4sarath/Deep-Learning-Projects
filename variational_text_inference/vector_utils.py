
import cPickle
import numpy as np

def vector_from_document_tfidf(tfidf_dict , jd_tokens, the_model):

    vec_count = 0
    vec_sum = 0
    for tok in jd_tokens:
        if tok in the_model.vocab:
            if tok in tfidf_dict:
                vec_count += 1
                vec_sum += tfidf_dict[tok]*the_model[tok]
            else:
                vec_count += 1
                vec_sum += the_model[tok]
    if vec_count > 0:
        vector = np.divide(vec_sum, float(vec_count))
        return vector

def vector_from_document_tfidf_dict(tfidf_dict , jd_tokens, the_model, vec_dim = 300):

    vec_count = 0
    vec_sum = 0
    for tok in jd_tokens:
        if tok in the_model:
            if tok in tfidf_dict:
                vec_count += 1
                vec_sum += tfidf_dict[tok]*the_model[tok]
            else:
                vec_count += 1
                vec_sum += the_model[tok]
    if vec_count > 0:
        vector = np.divide(vec_sum, float(vec_count))
        return vector
    else:
        return np.zeros(vec_dim)

def vector_from_document_tfidf_mongo(tfidf_dict , tokens, collections, vec_dim = 300):

    vec_count = 0
    vec_sum = 0
    for tok in tokens:
        res = collections.find_one({"word_name": tok})
        if res:
            vec_count += 1
            vec_sum += tfidf_dict[tok]*np.array(res[u'word_vec'])

    if vec_count > 0:
        vector = np.divide(vec_sum, float(vec_count))
        return vector
    else:
        return np.zeros(vec_dim)


def find_norm( syn0 ):
    syn0norm = (syn0 / np.sqrt((syn0 ** 2).sum(-1))[..., np.newaxis]).astype(np.float32)
    return syn0norm



def argsort(x, topn=None, reverse=False):
    """
    Return indices of the `topn` smallest elements in array `x`, in ascending order.

    If reverse is True, return the greatest elements instead, in descending order.

    """
    x = np.asarray(x)  # unify code path for when `x` is not a numpy array (list, tuple...)
    if topn is None:
        topn = x.size
    if topn <= 0:
        return []
    if reverse:
        x = -x
    if topn >= x.size or not hasattr(np, 'argpartition'):
        return np.argsort(x)[:topn]
    # numpy >= 1.8 has a fast partial argsort, use that!
    most_extreme = np.argpartition(x, topn)[:topn]
    return most_extreme.take(np.argsort(x.take(most_extreme)))  # resort topn into order



def find_similar(des_norm , vec_norm):

	dists = np.dot(des_norm , vec_norm)

	best = argsort(dists, reverse=True)

	dist_sort = np.sort( dists )[::-1]

	return  dist_sort , best


if __name__ == "__main__":
    pass

