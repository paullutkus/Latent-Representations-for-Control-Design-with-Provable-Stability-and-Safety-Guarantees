import numpy as np
import pickle
import params
import datetime 
from training import train
import matplotlib.pyplot as plt


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

    #print(exp)    
    
    if plot:
        plt.cla()
        for s in metrics:
            #rewards_config = experiment["rewards"]

            #colors = ['red', 'green', 'blue']
            #labels = ['backward-conjugate', 'forward-conjugate', 'both']
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 10)
            ax.set_title("{} vs training epoch".format(s))
            for i in range(n_combinations):
                configuration = exp[s][i]
                for j, run in enumerate(configuration[1]):
                    #if j == 0:
                    #    plt.plot(run, color=colors[i], label=labels[i])
                    #else:
                    ax.plot(run, label=configuration[0]+"\nrun {}".format(j+1))
            ax.legend(fontsize='xx-small')
            plt.show()

            #colors = ['red', 'green', 'blue']
            #labels = ['backward-conjugate', 'forward-conjugate', 'both']
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
        


def run_experiment(data, name=None, save=True, n_runs=[2, 2, 2]):
    X, U, Xtest, Utest = data

    experiment = {}

    ae_config              = []
    fdyn_config            = []
    ae_opt_config          = []
    fdyn_opt_config        = []
    ae_list_config         = []
    fdyn_list_config       = []
    rewards_config         = []
    completion_rate_config = []
    gamma_config           = []

    for i in range(len(params.penalize_rec_schedule)):
        params.penalize_rec = params.penalize_rec_schedule[i]
        params.predict_mstep = params.predict_mstep_schedule[i]
        params.penalize_reproj = params.penalize_reproj_schedule[i]
        params.penalize_encoder_diagram_mstep = params.penalize_encoder_diagram_mstep_schedule[i]
        ae_runs              = [] 
        fdyn_runs            = []
        ae_opt_runs          = []
        fdyn_opt_runs        = []
        ae_list_runs         = []
        fdyn_list_runs       = []
        rewards_runs         = []
        completion_rate_runs = []
        gamma_runs           = []

        for j in range(n_runs[i]):
            ae, fdyn, ae_opt, fdyn_opt, ae_list, fdyn_list, rewards_current_run, completion_rate_current_run, gamma_current_run = train(X, U, Xtest, Utest)
            ae_runs.append(ae)
            fdyn_runs.append(fdyn)
            ae_opt_runs.append(ae_opt)
            fdyn_opt_runs.append(fdyn_opt)
            ae_list_runs.append(ae_list)
            fdyn_list_runs.append(fdyn_list)
            rewards_runs.append(rewards_current_run)
            completion_rate_runs.append(completion_rate_current_run)
            gamma_runs.append(gamma_current_run)

        ae_config.append(ae_runs)
        fdyn_config.append(fdyn_runs)
        ae_opt_config.append(ae_opt_runs)
        fdyn_opt_config.append(fdyn_opt_runs)
        ae_list_config.append(ae_list_runs)
        fdyn_list_config.append(fdyn_list_runs)
        rewards_config.append(rewards_runs)
        completion_rate_config.append(completion_rate_runs)
        gamma_config.append(gamma_runs)

    experiment["ae"]              = ae_config
    experiment["fdyn"]            = fdyn_config
    experiment["ae_opt"]          = ae_opt_config
    experiment["fdyn_opt"]        = fdyn_opt_config
    experiment["ae_list"]         = ae_list_config
    experiment["fdyn_list"]       = fdyn_list_config
    experiment["rewards"]         = rewards_config
    experiment["completion_rate"] = completion_rate_config
    experiment["gamma"]           = gamma_config

    experiment["cfg"] = [(str(name), str(values)) for name, values in vars(params).items()]
    
    if name is None:
        name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    if save:
        save_experiment(name, experiment)

    return experiment


def save_experiment(name, experiment): #,ae_list_exp, fdyn_list_exp, rewards_exp, completion_rate_exp):
    #ae_list_exp = experiment
    #cfg = [(str(name), str(values)) for name, values in vars(params).items()]
    #experiment = [ae_list_exp, fdyn_list_exp, rewards_exp, completion_rate_exp, cfg] 
    print("saving {}".format(name + '.pkl'))
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(experiment, f)


def load_experiment(name):
    with open(name + '.pkl', 'rb') as f:
        experiment = pickle.load(f)
    return experiment


