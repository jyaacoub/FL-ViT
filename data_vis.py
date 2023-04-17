
# %%
import re, ast, os
import matplotlib.pyplot as plt

def get_metrics(path='media/32BS_e-5LR_1000Train_100Test/ConvNeXt_CIFAR10_5Clients_10Rounds_1000Train_100Test.ipynb'):
    """ Get metrics from a notebook file """
    with open(path, 'r') as f:
        doc = f.read()
        val_matched = re.findall(r".*metrics.*({'val_loss': \[\(.*})", doc)[0]
        test_matched = re.findall(r".*metrics.*({'test_loss': \[\(.*})", doc)[0]
        train_matched = re.findall(r"{.*train_loss': ([0-9]*.[0-9]*).* ([0-9]*.[0-9]*)", doc)
        time_elapsed = float(re.findall(r"FL finished in ([0-9]*.[0-9]*)", doc)[0])/60 # in minutes
    
    metrics = ast.literal_eval(val_matched) # returns dict with test_loss and test_accuracy
    metrics['test_loss'] = ast.literal_eval(test_matched)['test_loss']
    metrics['test_accuracy'] = ast.literal_eval(test_matched)['test_accuracy']
    train_loss = []
    train_acc = []
    for l, a in train_matched:
        try:
            train_loss.append(float(l))
            train_acc.append(float(a))
        except Exception as e:
            print('Error parsing train_loss and train_accuracy: ', l, a)
            print('train_matched: ', train_matched)
        
    metrics['train_loss'] = train_loss
    metrics['train_accuracy'] = train_acc
    
    metrics['time_elapsed'] = time_elapsed
    return metrics

def plot_metrics(metrics, key='accuracy', color=(0, 0, 0), model_name=None):
    # plotting test and train accuracy
    # test accuracy has an extra element because it is calculated before training
    plt.plot([i for i in range(1, len(metrics['test_'+key]))],
             [r[1] for r in metrics['test_'+key][1:]], 
             label=model_name+' (test)', color=color)
    
    # plotting train with fainter color
    plt.plot([i for i in range(1, len(metrics['train_'+key])+1)],
             [r for r in metrics['train_'+key]],
             label=model_name+' (train)', color=color, alpha=0.3)

def plot_all_models(metrics_path, dataset='CIFAR-100',key='accuracy', 
                    colors=[(0, 0, 0), (1,0,0), (0, 1, 0), (0, 0, 1)]): # for 4 models
    i=0
    for file in os.listdir(metrics_path):
        if not file.endswith('.ipynb'):
            continue
        path = os.path.join(metrics_path, file)
        plot_metrics(get_metrics(path), key=key,
                     color=colors[i], 
                     model_name=file.split('_')[0])
        i+=1
        
    # plotting the points
    # lr = re.findall(r"e-([0-9]*)LR", metrics_path)[0]
    plt.xlabel('Round')
    plt.ylabel(KEY.capitalize())
    # plt.title(f'{dataset} {KEY.capitalize()} with Federated Learning (1e-{lr}LR)')
    plt.title(f'{dataset} {KEY.capitalize()} with Federated Learning')
    # x = [i for i in range(1, 11)]
    # plt.xticks(x)
    # ledgend at top left
    plt.legend(loc='lower left')
    
    def no_overlap(y, prev_y, thresh=.05):
        if round(y,1) in prev_y:
            if prev_y[round(y,1)] > y:
                y -= thresh
            else:
                y += thresh
        else:
            prev_y[round(y,1)] = y
        return y, prev_y

    # plot time elapsed:
    # prev_y = {}
    # i=0
    # for file in os.listdir(metrics_path):
    #     if not file.endswith('.ipynb'):
    #         continue
    #     path = os.path.join(metrics_path, file)
    #     metrics = get_metrics(path)
    #     y_pos = metrics['test_'+key][-1][1]
    #     # making sure the text is not overlapping
    #     y_pos, prev_y = no_overlap(y_pos, prev_y)
    #     # if file[0] =='D': # hardcoded to ensure that the text is not overlapping (DeiT model)
    #     #     y_pos += .02
        
    #     num_rounds = len(metrics['test_'+key])
    #     x_pos = num_rounds + num_rounds/20
    #     plt.text(x_pos, y_pos, str(int(metrics['time_elapsed']))+' mins', color=colors[i])
    #     i+=1

# %%
METRICS_PATH = lambda x: f'media/hetero/1e-{x}LR/'
# METRICS_PATH = lambda x: f'media/hetero_real/'
# METRICS_PATH = lambda x: f'media/non-hetero/cifar100/'
KEY = 'accuracy'

# plt.figure(figsize=(10, 5))
# plot_all_models(METRICS_PATH(5), key=KEY)
# plt.savefig(METRICS_PATH(5)+KEY+'_all_models.png')

plt.figure(figsize=(10, 5))
plot_all_models(METRICS_PATH(4), dataset='CIFAR-100', key=KEY)
plt.savefig(METRICS_PATH(4)+KEY+'_all_models_notime.png')
plt.show()
# %%
