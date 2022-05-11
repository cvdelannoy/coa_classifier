from pathlib import Path

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from plot_coa_event_fingerprints import extract_fingerprints_from_files


def flatten_dict_to_x_y(event_dict):
    """Convert event_dict to format that can be used by KNN

    :param event_dict: Dictionary with per-coa event fingerprints. Output
                       by extract_fingerprints_from_files
    :return: x, which is list of tuples of (event_length, rel_blockade)
             y, which is ground cOA label as int such as 3 for coa3.
    """
    x = []
    y = []
    for abf_type, measurements in event_dict.items():
        abf_int = int(abf_type[-1])
        y.extend([abf_int] * len(measurements[0]))
        x.extend(zip(measurements[0], measurements[1]))
    return x, y

def fit_knn(x, y, k=5):
    """Train a k-nearest neighbour classifier on extracted data.

    :param x: list of (event_length, rel_blockade) observations per event
    :param y: ground truth of event
    :param k: number of nearest neighbours
    :return: KNN model
    :rtype: KNeighborsClassifier
    """
    print('Fitting KNN classifier')
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x, y)
    return neigh


def main():
    coa_folder = Path('/home/noord087/lustre_link/capita_selecta/data/120_mv_trans/train')
    coa_files = list(coa_folder.iterdir())
    event_dict = extract_fingerprints_from_files(coa_files)
    x_train, y = flatten_dict_to_x_y(event_dict)
    scaler = preprocessing.StandardScaler().fit(x_train)

    test_folder = Path('/home/noord087/lustre_link/capita_selecta/data/120_mv_trans/test')
    test_files = test_folder.iterdir()
    test_event_dict = extract_fingerprints_from_files(test_files)
    x_test, y_true = flatten_dict_to_x_y(test_event_dict)
    x_test = scaler.transform(x_test)
    x_train = scaler.transform(x_train)
    k = 3
    classifier = fit_knn(x_train, y, k)
    y_pred = classifier.predict(x_test)
    print(confusion_matrix(y_true, y_pred, normalize='true'))
    print(balanced_accuracy_score(y_true, y_pred))

    # for k in range(1, 20):
    #     classifier = fit_knn(x_train, y, k)
    #     y_pred = classifier.predict(x_test)
    #     # print(confusion_matrix(y_true, y_pred, normalize='true'))
    #     print('k is', k)
    #     print(balanced_accuracy_score(y_true, y_pred))
    # # plot_scatter_fingerprints(event_dict)

if __name__ == '__main__':
    main()