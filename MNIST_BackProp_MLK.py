import MachineLearningKit as mlk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def main():

    n_class = 10

    # Base de dados de treinamento
    print(f'Loading dataset')
    dataset = pd.read_csv('mnist_train_small.csv')
    n_inst = len(dataset.index) #500

    # Filtrando apenas números específicos
    # dataset = dataset.loc[dataset['7'] == 1]
    # dataset = dataset[dataset['6'].isin([1,4])]
    print(f'Adapting dataset')
    dataset = dataset.iloc[0:n_inst]
    dataset.iloc[:, 1:] = dataset.iloc[:, 1:] / 255
    # dataset.iloc[:, 1:] = dataset.iloc[:, 1:] * 2. - 1.

    print(f'Loading and adapting test dataset')
    test_dataset = pd.read_csv('mnist_test.csv')
    test_dataset = test_dataset.iloc[0:n_inst]
    test_dataset.iloc[:, 1:] = test_dataset.iloc[:, 1:] / 255
    #test_dataset.iloc[:, 1:] = test_dataset.iloc[:, 1:] * 2. - 1.

    X = dataset.iloc[:, 1:].to_numpy()

    # print(np.shape(X))

    y = [[0] * n_class] * n_inst

    for i in range(0, n_inst):
        y[i] = list(mlk.output_layer_activation(output_value=dataset.iloc[i, 0], num_classes=n_class))

    # print(np.shape(y))


    clf = mlk.load_neural_network(
        f'MNIST_BackProp last.xlsx')

    clf.max_iter = 200
    clf.tol = 1e-6
    clf.n_iter_no_change = 3
    clf.learning_rate_init = 5e-1
    clf.momentum=clf.learning_rate_init
    learning_rate = 'constant' #'invscaling'
    # clf = mlk.MLPClassifier(
    #     hidden_layer_sizes=((15)),
    #     activation=mlk.activation_function_name.TANH,
    #     learning_rate='invscaling',  # 'constant'
    #     solver=mlk.solver.BACKPROPAGATION,
    #     learning_rate_init=0.5,  # 0.001 para constant
    #     max_iter=10,
    #     shuffle=True,
    #     random_state=1,
    #     momentum=0.9,  # 0.01 para constant
    #     n_individuals=10,
    #     weight_limit=1,
    #     batch_size='auto',
    #     tol=0.01
    # )

    Eav, ne = clf.fit(X, y)
    print(f'Testando acertividade:')
    time = datetime.datetime.now()
    filename = f'MNIST_results ' \
               f'{time.year}-{time.month} - {time.day} ' \
               f'{time.hour}-{time.minute}-{time.second}.xlsx'

    mlk.teste_acertividade(X, y, clf, print_result=False,
                           save_result = True,
                           filename=filename)
    print(f'Épocas necessárias: {ne}')
    print(f'Acertividade:{clf.get_acertividade():.2f}%')
    plt.plot(Eav[0:(ne * n_inst)])
    plt.show()
    time = datetime.datetime.now()
    clf.save_neural_network(f'MNIST_BackProp {time.year}-{time.month}'
                            f'-{time.day} {time.hour}-{time.minute}'
                            f'-{time.second}.xlsx')
    clf.save_neural_network(f'MNIST_BackProp last.xlsx')

if __name__ == '__main__':
    main()