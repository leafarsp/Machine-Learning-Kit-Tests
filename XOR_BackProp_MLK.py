import MachineLearningKit as mlk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import MachineLearningCommonFunctions as mlkCF

def main():
    mlkCF.print_event_time('Start time')
    X, y, n_inst = prepare_dataset()

    clf = create_new_classifier()
    # clf = load_classifier('XOR_nn_obj.nn')
    # clf.max_epoch_sprint = clf.t + 200

    Eav, ne = clf.fit(X,y)
    # test_accuracy(X, y, clf)
    # clf.max_epoch_sprint = clf.t + 200

    # Eav, ne = clf.fit(X, y)
    test_accuracy(X, y, clf)





    print_results(X, y, clf, ne, n_inst, Eav)
    clf.save_nn_obj('XOR_nn_obj.nn')


def print_results(X, y, clf, ne, n_inst, Eav):
    print(f'Épocas necessárias: {ne}')
    plt.plot(Eav[0:(ne)])
    plt.show()
    plt_retas(clf, X, y)
    print(f'Acertividade:{clf.get_acertividade()}%')
    # print(a.get_output_class())
    # a.save_neural_network('teste.xlsx')
def test_accuracy(X_t, y_t, clf):
    print(f'Testando acertividade:')
    mlk.teste_acertividade(X_t, y_t, clf,
                           print_result=True,
                           save_result=True,
                           filename='XOR_results.xlsx')

def create_new_classifier():
    clf = mlk.MLPClassifier(
        hidden_layer_sizes=((2)),
        activation=mlk.activation_function_name.TANH,
        learning_rate='invscaling',  # 'constant'
        solver=mlk.solver.BACKPROPAGATION,
        learning_rate_init=0.5,  # 0.001 para constant
        max_iter=10000,
        shuffle=True,
        random_state=1,
        momentum=0.1,  # 0.01 para constant
        n_individuals=10,
        weight_limit=1,
        batch_size='auto',
        tol=1e-6,
        n_iter_no_change=10,


    )
    # clf.max_epoch_sprint=3000
    return clf

def load_classifier(filename):
    clf = mlk.load_nn_obj(filename)
    clf.flag_teste_acertividade = False
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
def plt_retas(rede,X, y):
    # Realiza construção do gráfico 2D das entradas e as retas
    num_inst = np.shape(X)[0]
    for n in range(0, num_inst):
        x1 = X[n, 0]
        x2 = X[n, 1]
        d = rede.get_output_class(y[n])
        plt.scatter(x1, x2, marker=f'${int(d)}$', s=200)
    x_space = np.linspace(-1, 1, 10)

    for j in range(0, rede.m[rede.L - 1]):
        b1 = rede.l[rede.L - 2].w[j][2]
        w1 = rede.l[rede.L - 2].w[j][0]
        w2 = rede.l[rede.L - 2].w[j][1]

        cy1 = (-b1 - w1 * x_space) / w2
        plt.plot(x_space, cy1)

    plt.show()

def test_convert_models(clf):
    clf.save_neural_network(f'XOR_before_scikit_converted.xlsx')

    clf_scikit = clf.convert_model_to_SciKitLearning()
    clf_scikit.warm_start = True
    mlk.load_scikit_model(clf_scikit).save_neural_network(f'XOR_scikit_converted.xlsx')
    clf_scikit.fit(X, y)

    mlkCF.teste_acertividade_Scikit(X, y, clf_scikit, True)

if __name__ == '__main__':
    main()