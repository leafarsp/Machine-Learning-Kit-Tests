import MachineLearningKit as mlk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import MachineLearningCommonFunctions as mlkCF

project_path = f'{mlkCF.get_project_root()}\\Machine-Learning-Kit-Tests'
project_name = f'MNIST_BackProp_Mlk_last_32_16.nn'


def train_neural_network(clf, X: list, y: list ):
    old_accuracy = 0
    mlkCF.print_event_time('Start time')

    # clf = load_existing_classifier()
    # old_accuracy = clf.get_acertividade()
    # #
    # # clf.learning_rate_init *= 0.5
    # eta, alpha = mlk.get_momentum_andLearning_rate(clf.t,clf.t+100, clf)
    # print(f'Learning rate: {eta}, momentum: {alpha}')
    # print(clf.t)
    # # exit()

    # X, y, n_inst = prepare_dataset()
    n_inst = np.shape(X)[0]
    Eav, ne = clf.fit(X, y)


    # save_classifier(clf)

    mlkCF.print_event_time('End time')

    return clf

def create_new_classifier():
    clf = mlk.MLPClassifier(
        hidden_layer_sizes=((16, 8)),
        activation=mlk.activation_function_name.TANH,
        learning_rate='invscaling',  # 'constant' invscaling #adaptive
        solver=mlk.solver.BACKPROPAGATION,
        learning_rate_init=5e-1,  # 0.001 para constant

        max_iter=5000,
        n_iter_no_change=3,
        shuffle=True,
        random_state=1,
        momentum=1e-1,  # 0.01 para constant
        n_individuals=10,
        weight_limit=1.,
        batch_size=1,
        tol=1e-6,
        activation_lower_value=0.
    )
    clf.max_epoch_sprint = 5000
    # clf.learning_rate_div=1.5
    # clf.power_t = 0.3
    return clf


def load_existing_classifier():
    # clf = mlk.load_neural_network(
    #     f'MNIST_BackProp last.xlsx')
    # clf = mlk.load_nn_obj(f'{project_path}\\Testes\\MNIST_BackProp_Mlk 2023-02-15 10-15-41.nn')
    clf = mlk.load_nn_obj(f'{project_path}\\{project_name}')
    clf.flag_teste_acertividade = False
    clf.max_epoch_sprint = clf.t + 200
    # clf.batch_size=50
    # clf.power_t = 0.05
    # # clf.tol = 1e-9
    # clf.momentum = 0.001
    # clf.cnt_error_free = 0
    # clf.learning_rate_changed = False

    # clf.momentum /= 3

    # clf.max_iter = 100
    # clf.tol = 1e-6
    # clf.n_iter_no_change = 3
    # clf.learning_rate_init = 1e-1
    # clf.momentum = 1e-1
    # clf.learning_rate = 'invscaling'  # 'invscaling' constant
    return clf


def save_classifier(clf: mlk.MLPClassifier):
    t = datetime.datetime.now()
    filename = f'{project_path}\\Testes\\MNIST_BackProp_Mlk ' \
               f'{t.year:02d}-{t.month:02d}-{t.day:02d} ' \
               f'{t.hour:02d}-{t.minute:02d}-{t.second:02d}'
    # clf.save_nn_obj(f'{project_path}\\Testes\\{project_name}')
    clf.save_nn_obj(f'{project_path}\\{project_name}')
    clf.save_nn_obj(f'{filename}.nn')
    clf.save_neural_network(f'{filename}.xlsx')


def test_accuracy(X_t, y_t, clf):
    print(f'Testando acertividade:')
    t = datetime.datetime.now()
    filename = f'{project_path}\\Testes\\MNIST_results ' \
               f'{t.year:02d}-{t.month:02d}-{t.day:02d} ' \
               f'{t.hour:02d}-{t.minute:02d}-{t.second:02d}.xlsx'
    clf.flag_teste_acertividade = False

    mlk.teste_acertividade(X_t, y_t, clf, print_result=False,
                           save_result=True,
                           filename=filename)
    print(f'Acertividade:{clf.get_acertividade():.2f}%')


def print_results(clf, ne, n_inst, Eav):
    print(f'Épocas necessárias: {ne}')
    plt.plot(Eav[0:(ne * n_inst)])
    plt.show()
    t = datetime.datetime.now()
    clf.save_neural_network(f'{project_path}\\Testes\\MNIST_BackProp {t.year}-{t.month}'
                            f'-{t.day} {t.hour}-{t.minute}'
                            f'-{t.second}.xlsx')
    clf.save_neural_network(f'{project_path}\\Testes\\MNIST_BackProp last.xlsx')



