import numpy as np
import pickle
import params
import datetime 
from training import train
import matplotlib.pyplot as plt


# train and save a set of models over a grid of parameters
def run_experiment_new(data, spec, fname=None, save=True, n_runs=[2,2,2], plot=True):
    if data is not None:
        X, U, Xtest, Utest = data
    # save original variables 
    original_vars = []
    for name in spec.keys():
        original_vars.append(eval("params.{}".format(name)))
    print("original state:\n", original_vars)

    n_combinations = len(list(spec.values())[0])
    n_params = len(spec.keys())
    print("n_combinations", n_combinations)
    print("n_params", n_params)
    exp = {}
    outputs = ["ae", "fdyn", "ae_opt", "fdyn_opt", "ae_list",
               "fdyn_list", "rewards", "completion_rate", "gamma"]
    metrics = ["rewards", "completion_rate", "gamma"]
    for o in outputs:
        configuration_names = []
        for i in range(n_combinations):
            configuration_names.append('\n'.join([list(spec.keys())[j]+'='+str(list(spec.values())[j][i]) for j in range(n_params)]))
        exp[o] = [(configuration_names[j], []) for j in range(n_combinations)]
    print(exp)           

    for i in range(n_combinations):
        for name, val in spec.items():
            exec('params.{} = {}'.format(name, val[i]))
        for j in range(n_runs[i]):
            ae, fdyn, ae_opt, fdyn_opt, ae_list, fdyn_list, rewards, completion_rate, gamma = train(X, U, Xtest, Utest)
            for o in outputs:
                # [o][i][0] is experiment details, [o][i][1] is values
                exp[o][i][1].append(eval(o))
    
    if plot:
        plt.cla()
        for s in metrics:
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 10)
            ax.set_title("{} vs training epoch".format(s))
            for i in range(n_combinations):
                configuration = exp[s][i]
                for j, run in enumerate(configuration[1]):
                    ax.plot(run, label=configuration[0]+"\nrun {}".format(j+1))
            ax.legend(fontsize='xx-small')
            plt.show()

            fig, ax = plt.subplots()
            fig.set_size_inches(10, 10)
            ax.set_title("{} vs training epoch (avg'd over {} runs)".format(s, len(exp[s][i][1])))
            for i in range(n_combinations):
                if len(exp[s][i][1]) > 0:
                    all_runs = exp[s][i][1]
                    min_len = min([len(run) for run in all_runs])
                    avg_run = np.array(all_runs[0][-min_len:]).astype(np.float64)
                    for run in all_runs[1:]:
                        avg_run += np.array(run[-min_len:]).astype(np.float64)
                    avg_run /= len(all_runs)
                    ax.plot(avg_run, label=exp[s][i][0])
            ax.legend(fontsize='xx-small')
            plt.show()

    # reset variables
    final_vars = []
    for name, val in zip(spec.keys(), original_vars):
        exec('params.{} = {}'.format(name, val))
        final_vars.append(eval('params.{}'.format(name)))
    print("final state\n", final_vars)

    if save:
        save_experiment(fname, exp)

    return exp
 

# save model and stats dict as '.pkl'
def save_experiment(name, experiment): 
    print("saving {}".format(name + '.pkl'))
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(experiment, f)


# load model and stats dict from '.pkl'
def load_experiment(name):
    with open(name + '.pkl', 'rb') as f:
        experiment = pickle.load(f)
    return experiment


