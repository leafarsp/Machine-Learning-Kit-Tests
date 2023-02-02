# import MachineLearningKit as mlk2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neural_network as mlk
import pickle
import datetime

def main():
    print_event_time('Start time')

    X, y, n_inst = prepare_dataset()
    # clf = create_new_classifier()

    clf = load_nn_obj('clf_scikit_XOR.nn')
    clf.warm_start=True

    clf.fit(X,y)

    test_accuracy(X, y, clf)


    # for i in range(0, len(clf.coefs_)):
    #     print(f'\n Layer {i}')
    #     print(f'Weights: {np.transpose(clf.coefs_[i])}')
    #     print(f'Bias:{clf.intercepts_[i]}')


    plt_retas(clf,X,y)

    save_nn_obj(clf, 'clf_scikit_XOR.nn')

    # print(a.get_output_class())
    # a.save_neural_network('teste.xlsx')
def save_nn_obj(obj, filename):
    with open(filename, 'wb') as outp:
        # Step 3
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
def load_nn_obj(filename):
    with open(filename, 'rb') as inp:
        clf = pickle.load(inp)
    return clf
def test_accuracy(X_t, y_t, clf):
    print(f'Testando acertividade:')
    acert = teste_acertividade(X_t, y_t, clf, print_result=True)
    print(f'Acertividade: {acert}%')

def create_new_classifier():
    clf= mlk.MLPClassifier(
        hidden_layer_sizes=((2)),
        activation= 'tanh',
        learning_rate = 'invscaling', # 'constant'
        solver = 'sgd',
        learning_rate_init = 0.5, # 0.001 para constant
        max_iter = 30,
        shuffle = True,
        random_state = 1,
        momentum=0.9, # 0.01 para constant

        batch_size='auto',
        tol=0.001,
        verbose = True,
        n_iter_no_change=10
    )
    return clf
def prepare_dataset():
    high_lim = 1.
    low_lim = 0.
    data = {'y1': [high_lim, low_lim, low_lim, high_lim],
            'y2': [low_lim, high_lim, high_lim, low_lim],
            'x1': [low_lim, low_lim, high_lim, high_lim],
            'x2': [low_lim, high_lim, low_lim, high_lim]}
    dataset = pd.DataFrame(data=data)
    n_inst = len(dataset.index)

    X = dataset.loc[:, ['x1', 'x2']].to_numpy()
    y = dataset.loc[:, ['y1', 'y2']].to_numpy()
    dataset.drop(columns=['y1'], inplace=True)
    print(dataset)
    return X,y,n_inst
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
def plt_retas(rede,X, y):
    # Realiza construção do gráfico 2D das entradas e as retas
    num_inst = np.shape(X)[0]
    for n in range(0, num_inst):
        x1 = X[n, 0]
        x2 = X[n, 1]
        d = get_output_class(y[n])
        plt.scatter(x1, x2, marker=f'${int(d)}$', s=200)
    x_space = np.linspace(0, 1, 10)

    layers = len(rede.coefs_)
    layer = layers-2


    weights = np.transpose(rede.coefs_[layer])
    bias = rede.intercepts_[layer]
    n_neuronios_layer = len(bias)
    # print(f'Weights: {weights}\n Bias: {bias}')
    # exit()

    for j in range(0, n_neuronios_layer):
        b1 = bias[j]

        w1 = weights[j][0]
        w2 = weights[j][1]

        cy1 = (-b1 - w1 * x_space) / w2
        plt.plot(x_space, cy1)

    plt.show()

def teste_acertividade(X: list, y: list, rede: mlk.MLPClassifier, print_result=False):
    cont_acert = 0
    wrong_text = ' - wrong'
    n_inst = np.shape(X)[0]

    for i in range(0, n_inst):

        num_real = get_output_class(y[i])
        y_l = rede.predict([X[i]])[0]
        num_rede = get_output_class(rede.predict([X[i]])[0])

        if num_rede != np.nan:

            if (num_real == num_rede):
                cont_acert += 1
                wrong_text = ""

        if print_result:
            print(f'Núm. real: {num_real}, núm rede: {num_rede}{wrong_text}, neurônios: {y_l}')
        wrong_text = ' - wrong'
    result = 100 * cont_acert / n_inst
    return result

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

if __name__ == '__main__':
    main()