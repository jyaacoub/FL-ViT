#%% all pretrained
viT = [(0, 0.16), (1, 0.22), (2, 0.26), (3, 0.42), (4, 0.48), (5, 0.58), (6, 0.62), (7, 0.76), (8, 0.68), (9, 0.82), (10, 0.78)]
ConvNext = [(0, 0.16), (1, 0.16), (2, 0.18), (3, 0.18), (4, 0.18), (5, 0.24), (6, 0.28), (7, 0.38), (8, 0.38), (9, 0.42), (10, 0.46)]
bit = [(0, 0.04), (1, 0.38), (2, 0.48), (3, 0.54), (4, 0.38), (5, 0.5), (6, 0.4), (7, 0.46), (8, 0.4), (9, 0.6), (10, 0.54)]
deit = [(0, 0.12), (1, 0.16), (2, 0.32), (3, 0.46), (4, 0.5), (5, 0.56), (6, 0.54), (7, 0.62), (8, 0.58), (9, 0.6), (10, 0.58)]

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
# plotting the points
plt.plot([r[0] for r in viT], [r[1] for r in viT], label = "ViT-B/16")
plt.plot([r[0] for r in deit], [r[1] for r in deit], label = "DeiT-B/16")
plt.plot([r[0] for r in bit], [r[1] for r in bit], label = "BiT-50")
plt.plot([r[0] for r in ConvNext], [r[1] for r in ConvNext], label = "ConvNeXt-tiny")
plt.xlabel('Round')
plt.ylabel('Accuracy (Centralized)')
plt.title('CIFAR-10 Accuracy with Federated Learning')


plt.legend()
plt.savefig('./media/centralized_accuracy.png')

# %%
