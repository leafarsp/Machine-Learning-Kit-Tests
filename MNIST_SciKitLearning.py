from sklearn.neural_network import MLPClassifier
import pandas as pd
import MachineLearningKit as mlk
import MachineLearningCommonFunctions as mlkCF


def main():
    mlkCF.print_event_time('Start time')

    X, y, n_inst = prepare_dataset()

    # clf = create_new_classifier()
    clf = load_existing_classifier()


    clf.fit(X, y)

    X_t, y_t, n_inst_t = prepare_test_dataset()
    clf_mlk = mlk.load_scikit_model(clf)
    test_accuracy(X_t, y_t, clf_mlk)

    mlkCF.save_classifier_scikit(clf)

    clf_mlk.save_neural_network(f'MNIST Scikit learn '
                                f'model converted.xlsx')

    mlkCF.save_nn_obj(clf_mlk, 'MNIST_BackProp_Scikit_converted.nn')

    mlkCF.print_event_time('End time')
def test_accuracy(X_t, y_t, clf):
    print(f'Testando acertividade:')

    acert = mlk.teste_acertividade(X_t, y_t, clf,
                                   save_result=True,
                                   filename=f'MNIST '
                                            f'scikitlearn '
                                            f'results.xlsx')
    print(f'Acertividade: {acert}%')
def load_existing_classifier():
    clf = mlkCF.load_nn_obj('MNIST_BackProp_SKL_last.nn')
    clf.warm_start = True
    # clf.learning_rate_init = 0.05
    # clf.momentum = 0.01
    clf.max_iter = 1000
    # clf.learning_rate = 'adaptive'
    return clf
def create_new_classifier():
    clf = MLPClassifier(
        hidden_layer_sizes=((32,16)),
        activation='tanh',
        learning_rate='adaptive',  # 'adaptive' ,#'invscaling',  # 'constant'
        solver='sgd',
        learning_rate_init=0.9,  # 0.001 para constant
        max_iter=40000,
        shuffle=True,
        random_state=1,
        momentum=0.1,  # 0.01 para constant

        batch_size='auto',
        tol=1e-8,
        verbose=True,
        n_iter_no_change=10,
        alpha=1e-9,
    )
    return clf

def prepare_dataset():
    n_class = 10

    # Base de dados de treinamento
    print(f'Loading dataset')
    dataset = pd.read_csv('mnist_train_small.csv')
    n_inst = len(dataset.index)  # 500

    # Filtrando apenas números específicos
    # dataset = dataset.loc[dataset['7'] == 1]
    # dataset = dataset[dataset['6'].isin([1,4])]
    print(f'Adapting dataset')
    dataset = dataset.iloc[0:n_inst]
    # dataset.iloc[:, 1:] = dataset.iloc[:, 1:] / 255
    dataset[dataset.columns[1:]]=dataset.iloc[:, 1:] / 255

    # dataset.iloc[:, 1:] = dataset.iloc[:, 1:] * 2. - 1.



    X = dataset.iloc[:, 1:].to_numpy()

    # print(np.shape(X))

    y = [[0] * n_class] * n_inst

    for i in range(0, n_inst):
        y[i] = list(mlk.
        output_layer_activation(
            output_value=dataset.iloc[i, 0],
            num_classes=n_class,
            activation_lower_value=0.))
    return X,y, n_inst


def prepare_test_dataset():
    n_class = 10


    print(f'Loading and adapting test dataset')
    test_dataset = pd.read_csv('mnist_test.csv')
    n_inst = len(test_dataset.index)  # 500
    test_dataset = test_dataset.iloc[0:n_inst]
    test_dataset[test_dataset.columns[1:]] = test_dataset.iloc[:, 1:] / 255
    # test_dataset.iloc[:, 1:] = test_dataset.iloc[:, 1:] / 255
    # test_dataset.iloc[:, 1:] = test_dataset.iloc[:, 1:] * 2. - 1.

    X = test_dataset.iloc[:, 1:].to_numpy()

    # print(np.shape(X))

    y = [[0] * n_class] * n_inst

    for i in range(0, n_inst):
        y[i] = list(mlk.
        output_layer_activation(
            output_value=test_dataset.iloc[i, 0],
            num_classes=n_class,
            activation_lower_value=0.))
    return X,y, n_inst



if __name__ == '__main__':
    main()


