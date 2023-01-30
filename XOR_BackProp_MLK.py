import MachineLearningKit as mlk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
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

    clf=mlk.MLPClassifier(
        hidden_layer_sizes=((2)),
        activation= mlk.activation_function_name.TANH,
        learning_rate = 'invscaling', # 'constant'
        solver = mlk.solver.BACKPROPAGATION,
        learning_rate_init = 0.5, # 0.001 para constant
        max_iter = 30000,
        shuffle = True,
        random_state = 1,
        momentum=0.9, # 0.01 para constant
        n_individuals = 10,
        weight_limit=1,
        batch_size='auto',
        tol=0.00001,
        n_iter_no_change=10
    )

    Eav, ne = clf.fit(X,y)
    print(f'Testando acertividade:')
    mlk.teste_acertividade(X, y, clf,
                           print_result=False,
                           save_result=True,
                           filename='XOR_results.xlsx')
    print(f'Épocas necessárias: {ne}')
    plt.plot(Eav[0:(ne)])
    plt.show()
    plt_retas(clf, X, y)
    print(f'Acertividade:{clf.get_acertividade()}%')
    # print(a.get_output_class())
    # a.save_neural_network('teste.xlsx')

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

if __name__ == '__main__':
    main()