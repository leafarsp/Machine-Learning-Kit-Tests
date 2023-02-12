import datetime
import numpy as np
import sklearn.neural_network as skl
import pickle
import pandas as pd




def print_event_time(str_event):
    t = datetime.datetime.now()
    print(f'{str_event} {t.year:04d}-{t.month:02d}-{t.day:02d} - '
          f'{t.hour:02d}:{t.minute:02d}:{t.second:02d}')

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

def teste_acertividade(X: list, y: list, rede: skl,
                       filename, print_result=False):
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
            print(f'Núm. real: {num_real}, '
                  f'núm rede: {num_rede}{wrong_text}, '
                  f'neurônios: {y_l}')
        wrong_text = ' - wrong'
    result = 100 * cont_acert / n_inst
    list_row = ['', '', 'Accuracy', f'{result:.2f}%']
    list_row += [''] * len(y[0])
    df.loc[len(df)] = list_row
    df.to_excel(filename, sheet_name='Results')

    return result


def save_nn_obj(obj, filename):
    with open(filename, 'wb') as outp:
        # Step 3
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def cast_to_MLP_classifier(clf:skl.MLPClassifier):
    return clf

def load_nn_obj(filename):
    with open(filename, 'rb') as inp:
        clf = pickle.load(inp)
    return cast_to_MLP_classifier(clf)

def output_layer_activation(output_value, num_classes,
                            lower_value=-1.):
    d = np.ones(num_classes, dtype=np.float64) * lower_value
    # num = dataset_shufle.iloc[ni, 0]
    d[output_value] = 1.
    return d
def save_classifier_scikit(clf:skl.MLPClassifier):
    t = datetime.datetime.now()
    filename = f'MNIST_BackProp_SKL ' \
               f'{t.year:02d}-{t.month:02d}-{t.day:02d} ' \
               f'{t.hour:02d}-{t.minute:02d}-{t.second:02d}'
    save_nn_obj(clf, f'{filename}.nn')
    save_nn_obj(clf, f'MNIST_BackProp_SKL_last.nn')
