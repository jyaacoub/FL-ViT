
# %%
import re
with open('colab/media/ConvNeXt_CIFAR10_5Clients_10Rounds_1000Train_100Test.ipynb', 'r') as f:
    matched = re.findall(r"{.*train_loss': ([0-9]*.[0-9]*).* ([0-9]*.[0-9]*)", f.read())

loss = []
acc = []
for l, a in matched:
    loss.append(float(l))
    acc.append(float(a))

print(loss)
print(acc)


#%% all pretrained
ViT_metrics = {'val_loss': [(1, 2.2656664848327637), (2, 2.103411555290222), (3, 1.9848032593727112), (4, 1.8796334862709045), (5, 1.8450356721878052), (6, 1.6238410472869873), (7, 1.5225083827972412), (8, 1.5228939652442932), (9, 1.3764183521270752), (10, 1.3031120896339417)], 
           'val_accuracy': [(1, 0.275), (2, 0.3), (3, 0.35), (4, 0.45), (5, 0.4), (6, 0.625), (7, 0.6499999999999999), (8, 0.65), (9, 0.775), (10, 0.8)],
           'test_loss': [(0, 9.611557960510254), (1, 9.386661529541016), (2, 8.441346168518066), (3, 8.199491500854492), (4, 7.691163063049316), (5, 7.541686058044434), (6, 6.606864929199219), (7, 6.381008148193359), (8, 5.811273574829102), (9, 5.535859107971191), (10, 5.097431182861328)], 
           'test_accuracy': [(0, 0.11), (1, 0.12), (2, 0.21), (3, 0.28), (4, 0.33), (5, 0.45), (6, 0.54), (7, 0.66), (8, 0.68), (9, 0.74), (10, 0.8)],
           'train_loss': [2.2012189626693726, 2.0849703550338745, 1.9034101366996765, 2.0199947357177734, 1.6098512411117554, 1.4160170555114746, 1.4084084033966064, 1.3913161754608154, 1.1565086841583252, 1.0950335264205933],
           'train_accuracy': [0.18055555555555558, 0.1611111111111111, 0.3138888888888889, 0.4138888888888889, 0.5833333333333334, 0.6999999999999998, 0.7444444444444445, 0.8277777777777778, 0.8972222222222223, 0.8666666666666667]
           }

deit_metrics = {'val_loss': [(1, 2.1279748678207397), (2, 1.9230936169624329), (3, 1.838880717754364), (4, 1.8043901920318604), (5, 1.6851754784584045), (6, 1.4709340333938599), (7, 1.4722465872764587), (8, 1.3834007382392883), (9, 1.1327391862869263), (10, 1.0353561639785767)], 
            'val_accuracy': [(1, 0.32499999999999996), (2, 0.525), (3, 0.65), (4, 0.55), (5, 0.6), (6, 0.725), (7, 0.7), (8, 0.7), (9, 0.825), (10, 0.8)],
            'test_loss': [(0, 8.997632026672363), (1, 8.717939376831055), (2, 7.932766437530518), (3, 7.488421440124512), (4, 7.522444725036621), (5, 6.825922966003418), (6, 6.370735168457031), (7, 5.498552322387695), (8, 5.295784950256348), (9, 4.823629379272461), (10, 4.124497413635254)], 
            'test_accuracy': [(0, 0.2), (1, 0.25), (2, 0.41), (3, 0.61), (4, 0.62), (5, 0.64), (6, 0.69), (7, 0.73), (8, 0.77), (9, 0.77), (10, 0.79)],
            'train_loss': [2.1899110078811646, 2.155806303024292, 1.9227261543273926, 1.7074791193008423, 1.649557888507843, 1.4784363508224487, 1.4227023124694824, 1.2522885203361511, 1.0261735320091248, 0.9613142609596252],
            'train_accuracy': [0.19166666666666665, 0.3361111111111111, 0.4388888888888889, 0.6027777777777777, 0.6888888888888888, 0.6833333333333332, 0.763888888888889, 0.8083333333333332, 0.8083333333333332, 0.8611111111111112]
            }
bit_metrics  = {'val_loss': [(1, 2.120806872844696), (2, 1.371188998222351), (3, 1.0730488002300262), (4, 0.8550170361995697), (5, 0.993678867816925), (6, 0.7147002518177032), (7, 0.5338004529476166), (8, 0.5732885897159576), (9, 0.6396512389183044), (10, 0.5056430697441101)], 
            'val_accuracy': [(1, 0.42500000000000004), (2, 0.675), (3, 0.725), (4, 0.825), (5, 0.725), (6, 0.825), (7, 0.825), (8, 0.825), (9, 0.775), (10, 0.825)],
            'test_loss': [(0, 12.957881927490234), (1, 8.028347969055176), (2, 6.448009967803955), (3, 4.717401504516602), (4, 3.7986607551574707), (5, 2.797112464904785), (6, 3.1313672065734863), (7, 2.286500930786133), (8, 2.0153019428253174), (9, 1.6876938343048096), (10, 1.7226805686950684)], 
            'test_accuracy': [(0, 0.1), (1, 0.31), (2, 0.52), (3, 0.68), (4, 0.77), (5, 0.84), (6, 0.81), (7, 0.85), (8, 0.83), (9, 0.87), (10, 0.88)],
            'train_loss': [2.596365809440613, 1.7737002968788147, 1.1238199174404144, 0.8756062984466553, 0.6362078785896301, 0.6677687168121338, 0.37092435359954834, 0.18792754411697388, 0.3190924748778343, 0.32914717122912407],
            'train_accuracy': [0.14722222222222223, 0.4527777777777778, 0.6416666666666666, 0.763888888888889, 0.8305555555555556, 0.9083333333333333, 0.9305555555555555, 0.9611111111111109, 0.9277777777777777, 0.8972222222222223]
            }
cnxt_metrics = {'val_loss': [(1, 2.293596863746643), (2, 2.2903807163238525), (3, 2.269240975379944), (4, 2.240424871444702), (5, 2.244856595993042), (6, 2.242040991783142), (7, 2.2195680141448975), (8, 2.1843336820602417), (9, 2.18130362033844), (10, 2.1570236682891846)], 
            'val_accuracy': [(1, 0.125), (2, 0.1), (3, 0.15000000000000002), (4, 0.175), (5, 0.225), (6, 0.15000000000000002), (7, 0.175), (8, 0.3), (9, 0.22499999999999998), (10, 0.35)],
            'test_loss': [(0, 9.2679443359375), (1, 9.212156295776367), (2, 9.078756332397461), (3, 9.007659912109375), (4, 8.949664115905762), (5, 9.015214920043945), (6, 8.899678230285645), (7, 8.80817699432373), (8, 8.686306953430176), (9, 8.694424629211426), (10, 8.626787185668945)], 
            'test_accuracy': [(0, 0.07), (1, 0.08), (2, 0.09), (3, 0.11), (4, 0.14), (5, 0.17), (6, 0.18), (7, 0.22), (8, 0.26), (9, 0.3), (10, 0.33)],
            'train_loss': [2.3092198371887207, 2.310650587081909, 2.2789759635925293, 2.264200806617737, 2.2326483726501465, 2.245592713356018, 2.185489296913147, 2.213523745536804, 2.179749011993408, 2.1055840253829956],
            'train_accuracy': [0.05833333333333333, 0.10555555555555556, 0.09444444444444444, 0.13333333333333333, 0.14444444444444446, 0.2027777777777778, 0.23611111111111108, 0.27499999999999997, 0.36388888888888893, 0.4138888888888889]
            }
# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))

def plot_metrics(metrics, color=(0, 0, 0), model_name=None):
    # plotting test and train accuracy
    plt.plot([i for i in range(len(metrics['test_accuracy']))],
             [r[1] for r in metrics['test_accuracy']], 
             label=model_name+' (test)', color=color)
    
    # plotting train with fainter color
    plt.plot([i for i in range(len(metrics['train_accuracy']))],
             [r for r in metrics['train_accuracy']],
             label=model_name+' (train)', color=color, alpha=0.3)

plot_metrics(ViT_metrics, color=(0, 0, 0), model_name='ViT')
plot_metrics(deit_metrics, color=(1, 0, 0), model_name='DeiT')
plot_metrics(bit_metrics, color=(0, 1, 0), model_name='BiT')
plot_metrics(cnxt_metrics, color=(0, 0, 1), model_name='ConvNetXt')
# plotting the points
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('CIFAR-10 Accuracy with Federated Learning')

plt.legend()
# plt.savefig('./media/train_test_accuracy.png')
plt.show()
