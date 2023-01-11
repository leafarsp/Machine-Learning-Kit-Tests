import MachineLearningKit as mlk
import numpy as np
import pandas as pd

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

    a=mlk.MLPClassifier(
        hidden_layer_sizes=((10)),
        activation= mlk.activation_function_name.TANH,
        learning_rate = 'constant',
        solver = mlk.solver.BACKPROPAGATION,
        learning_rate_init = 0.001,
        max_iter = 200,
        shuffle = True,
        random_state = 1,
        n_individuals = 10
    )
    print(X[0])
    a.initialize_layers(2, 2)
    a.initialize_weights_random()
    eta = [0.8, 0.8]
    learning_rate_end = 0.01
    alpha = [0.1, 0.1]
    try:

        for i in range(0,4):
            a.forward_propagation(X[i])
            print(f'Saída antes do back propagation{a.l[1].y}')
            a.backward_propagation(X[i], y[i], alpha, eta)
            print(f'Saída depois do back propagation{a.l[1].y}\n')
    except ValueError as err:
        print(f'Erro: {err.args}')

    print(np.shape(a.l[1].w))

    # print(a.get_output_class())
    # a.save_neural_network('teste.xlsx')

if __name__ == '__main__':
    main()