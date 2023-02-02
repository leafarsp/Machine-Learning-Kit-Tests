from sklearn.neural_network import MLPClassifier
import pandas as pd
import matplotlib.pyplot as plt
# import classe_rede_neural as nnc
import numpy as np
import MachineLearningKit as mlk
import datetime
import pickle


def main():
    print_event_time('Start time')

    X, y, n_inst = prepare_dataset()

    clf = create_new_classifier()

    clf.fit(X, y)

    X_t, y_t, n_inst_t = prepare_test_dataset()
    clf_mlk = mlk.load_scikit_model(clf)
    test_accuracy(X_t, y_t, clf)

    clf_mlk.save_neural_network(f'MNIST Scikit learn '
                                f'model converted.xlsx')
    save_nn_obj(clf, 'MNIST_BackProp_Scikit.nn')

    print_event_time('End time')
def test_accuracy(X_t, y_t, clf):
    print(f'Testando acertividade:')

    acert = mlk.teste_acertividade(X_t, y_t, clf_mlk,
                                   save_result=True,
                                   filename=f'MNIST '
                                            f'scikitlearn '
                                            f'results.xlsx')
    print(f'Acertividade: {acert}%')
def create_new_classifier():
    clf = MLPClassifier(
        hidden_layer_sizes=((15,)),
        activation='tanh',
        learning_rate='invscaling',  # 'adaptive' ,#'invscaling',  # 'constant'
        solver='sgd',
        learning_rate_init=0.5,  # 0.001 para constant
        max_iter=200000,
        shuffle=True,
        random_state=1,
        momentum=0.5,  # 0.01 para constant

        batch_size='auto',
        tol=1e-8,
        verbose=True,
        n_iter_no_change=10,
        alpha=1e-5,
    )
    return clf
def print_event_time(str_event):
    t = datetime.datetime.now()
    print(f'{str_event} {t.year:04d}-{t.month:02d}-{t.day:02d} - '
          f'{t.hour:02d}:{t.minute:02d}:{t.second:02d}')
def save_nn_obj(obj, filename):
    with open(filename, 'wb') as outp:
        # Step 3
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
def load_nn_obj(filename):
    with open(filename, 'rb') as inp:
        clf = pickle.load(inp)
    return clf
def output_layer_activation(output_value, num_classes,
                            lower_value=-1.):
    d = np.ones(num_classes, dtype=np.float64) * lower_value
    # num = dataset_shufle.iloc[ni, 0]
    d[output_value] = 1.
    return d


def get_output_class(y, threshold=0.8):
    num_out = np.nan
    cont_neuronio_ativo = 0
    y_l = y

    for j in range(0, len(y_l)):
        if y_l[j] > (1 * threshold):
            num_out = j
            cont_neuronio_ativo += 1
        if cont_neuronio_ativo > 1:
            num_out = np.nan
            break
    return num_out

def teste_acertividade(X: list, y: list, rede: MLPClassifier, print_result=False):
    cont_acert = 0
    wrong_text = ' - wrong'
    n_inst = np.shape(X)[0]
    columns = ['Instance',
               'Real class',
               'Predicted class',
               'Result']

    columns += list(np.arange(len(y[0])))
    df = pd.DataFrame(columns=columns)

    for i in range(0, n_inst):

        num_real = get_output_class(y[i])
        y_l = rede.predict([X[i]])[0]
        num_rede = get_output_class(rede.predict([X[i]])[0])
        comparision_result = False
        if num_rede != np.nan:

            if (num_real == num_rede):
                cont_acert += 1
                comparision_result = True
                wrong_text = ""
        list_row = [i, num_real, num_rede, comparision_result]

        list_row = list(list_row) + list(y_l)
        df.loc[len(df)] = list_row
        if print_result:
            print(f'Núm. real: {num_real}, núm rede: {num_rede}{wrong_text}, neurônios: {y_l}')
        wrong_text = ' - wrong'
    result = 100 * cont_acert / n_inst
    list_row = ['', '', 'Accuracy', f'{result:.2f}%']
    list_row += [''] * len(y[0])
    df.loc[len(df)] = list_row
    df.to_excel(f'MNIST scikitlearn results.xlsx', sheet_name='Results')

    return result

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
    dataset.iloc[:, 1:] = dataset.iloc[:, 1:] / 255
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
    test_dataset.iloc[:, 1:] = test_dataset.iloc[:, 1:] / 255
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


