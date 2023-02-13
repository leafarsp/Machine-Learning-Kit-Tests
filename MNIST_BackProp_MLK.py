import MachineLearningKit as mlk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import MachineLearningCommonFunctions as mlkCF

project_path = f'{mlkCF.get_project_root()}\\Machine-Learning-Kit-Tests'

def main():

    mlkCF.print_event_time('Start time')

    X, y, n_inst = prepare_dataset()

    clf = load_existing_classifier()





    # clf.power_t = 0.15 # 0.1 feito a partir de t=1100
    clf = create_new_classifier()

    Eav, ne = clf.fit(X, y)

    X_t, y_t, n_inst_t = prepare_test_dataset()
    test_accuracy(X_t, y_t, clf)

    print_results(clf, ne, n_inst, Eav)

    save_classifier(clf)

    mlkCF.print_event_time('End time')


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

def create_new_classifier():
    clf = mlk.MLPClassifier(
        hidden_layer_sizes=((32,16)),
        activation=mlk.activation_function_name.TANH,
        learning_rate='adaptive',  # 'constant' invscaling #adaptive
        solver=mlk.solver.BACKPROPAGATION,
        learning_rate_init=9e-1,  # 0.001 para constant

        max_iter=10000,
        n_iter_no_change=3,
        shuffle=True,
        random_state=1,
        momentum=1e-1,  # 0.01 para constant
        n_individuals=10,
        weight_limit=1.,
        batch_size=1,
        tol=1e-4,
        activation_lower_value=0.
    )
    clf.max_epoch_sprint = 2000
    clf.learning_rate_div=1.5
    return clf

def load_existing_classifier():
    # clf = mlk.load_neural_network(
    #     f'MNIST_BackProp last.xlsx')
    clf = mlk.load_nn_obj(f'{project_path}\\Testes\\MNIST_BackProp_Mlk_last.nn')
    clf.flag_teste_acertividade=False
    clf.max_epoch_sprint = clf.t + 2000
    clf.batch_size=1


    # clf.max_iter = 100
    # clf.tol = 1e-6
    # clf.n_iter_no_change = 3
    # clf.learning_rate_init = 1e-1
    # clf.momentum = 1e-1
    # clf.learning_rate = 'invscaling'  # 'invscaling' constant
    return clf

def save_classifier(clf:mlk.MLPClassifier):
    t = datetime.datetime.now()
    filename = f'{project_path}\\Testes\\MNIST_BackProp_Mlk ' \
               f'{t.year:02d}-{t.month:02d}-{t.day:02d} ' \
               f'{t.hour:02d}-{t.minute:02d}-{t.second:02d}'
    clf.save_nn_obj(f'{project_path}\\Testes\\MNIST_BackProp_Mlk_last.nn')
    clf.save_nn_obj(f'{filename}.nn')
    clf.save_neural_network(f'{filename}.xlsx')

def prepare_dataset():
    n_class = 10

    # Base de dados de treinamento
    print(f'Loading dataset')
    dataset = pd.read_csv('mnist_train_small.csv')


    # Filtrando apenas números específicos
    # dataset = dataset.loc[dataset['3'] == 1]
    # dataset = dataset[dataset['6'].isin([1,4])]

    n_inst = 100#len(dataset.index)
    print(f'Adapting dataset')
    dataset = dataset.iloc[0:n_inst]
    # dataset.iloc[:, 1:] = dataset.iloc[:, 1:] / 255
    dataset[dataset.columns[1:]] = dataset.iloc[:, 1:] / 255
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



    return X, y, n_inst



def print_results(clf, ne, n_inst, Eav):
    print(f'Épocas necessárias: {ne}')
    plt.plot(Eav[0:(ne * n_inst)])
    plt.show()
    t = datetime.datetime.now()
    clf.save_neural_network(f'{project_path}\\Testes\\MNIST_BackProp {t.year}-{t.month}'
                            f'-{t.day} {t.hour}-{t.minute}'
                            f'-{t.second}.xlsx')
    clf.save_neural_network(f'{project_path}\\Testes\\MNIST_BackProp last.xlsx')


def prepare_test_dataset():
    n_class = 10

    print(f'Loading and adapting test dataset')
    test_dataset = pd.read_csv('mnist_test.csv')
    n_inst = len(test_dataset.index)  # 500
    test_dataset = test_dataset.iloc[0:n_inst]
    # test_dataset.iloc[:, 1:] = test_dataset.iloc[:, 1:] / 255
    test_dataset[test_dataset.columns[1:]] = test_dataset.iloc[:, 1:] / 255
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
    pd.DataFrame(X).to_csv('mnist_test_X.csv')
    pd.DataFrame(y).to_csv('mnist_test_y.csv')
    return X, y, n_inst

if __name__ == '__main__':
    main()