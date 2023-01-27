import MachineLearningKit as mlk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    high_lim = 1.
    low_lim = -1.
    data = {'y1': [high_lim, low_lim, low_lim, high_lim],
            'y2': [low_lim, high_lim, high_lim, low_lim],
            'x1': [-1., -1., 1., 1.],
            'x2': [-1., 1., -1., 1.]}
    dataset = pd.DataFrame(data=data)
    n_inst = len(dataset.index)

    X = dataset.loc[:, ['x1', 'x2']].to_numpy()
    y = dataset.loc[:, ['y1', 'y2']].to_numpy()

    dataset.drop(columns=['y1'], inplace=True)
    print(dataset)

    clf=mlk.MLPClassifier(
        hidden_layer_sizes=((10)),
        activation= mlk.activation_function_name.TANH,
        learning_rate = 'constant',
        solver = mlk.solver.BACKPROPAGATION,
        learning_rate_init = 0.001,
        max_iter = 20000,
        shuffle = True,
        random_state = 1,
        momentum=0.01,
        n_individuals = 10,
        weight_limit=1,
        batch_size='auto',
        tol=0.0001
    )

    Eav = clf.fit(X,y)
    mlk.teste_acertividade(X, y, clf, print_result=True)

    plt.plot(Eav)
    plt.show()
    # print(a.get_output_class())
    # a.save_neural_network('teste.xlsx')

if __name__ == '__main__':
    main()