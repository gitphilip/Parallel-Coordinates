import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from sklearn.preprocessing import StandardScaler

def parallel_coordinates(instances, labels,
                         class_names=[], feature_names=None,
                         show_mean_instances=False,
                         show_grid=True,
                         n_samples=None,
                         original_values=False,
                         save=False):
    
    scaler = StandardScaler()
    scaler.fit(instances)
    
    if n_samples is not None:
        idx = np.random.choice(instances.shape[0], 
                               n_samples, 
                               replace=False)
        instances = instances[idx, :]
        labels = labels[idx]
    
    X = scaler.transform(instances)
    
    colors = plt.get_cmap('Set2')
    
    if show_grid:
        plt.rcParams['grid.alpha'] = 0.3
        plt.grid(True)
    
    for i, instance in enumerate(X):
        plt.plot(instance, 
                 color=colors(labels[i]),
                 marker="o",
                 alpha=0.4)
    
    if original_values:
        ax = plt.gca()
        ax.set_xticks(np.arange(0, X.shape[1], 1))
        offset = 0.16
        for xloc in range(X.shape[1]):
            ymin = np.nanmin(instances[:, xloc])
            ymax = np.nanmax(instances[:, xloc])
            locs = [ymin, ymax]

            loc_matrix = np.array([locs,]*X.shape[1]).T
            scaled_locs = scaler.transform(loc_matrix)
            scaled_locs[0, :] -= offset
            scaled_locs[1, :] += offset
            for loc, scaled_loc in zip(locs, scaled_locs[:, xloc]):
                ax.text(xloc, scaled_loc, '%1.1f'%loc,
                    verticalalignment='center',
                    horizontalalignment='center')

    if show_mean_instances:
        for class_nr in np.unique(labels):
            mean_class_instance = np.nanmean(
                instances[labels == class_nr], 
                axis=0)
            mean_class_instance = scaler.transform(
                mean_class_instance.reshape(1,-1))[0]
            
            plt.plot(mean_class_instance,
                     color=colors(class_nr),
                     linewidth=3,
                     marker="o",
                     path_effects=[
                         pe.Stroke(linewidth=5, 
                                   foreground="black"), 
                         pe.Normal()])
        
    if feature_names is not None:
        plt.xticks(range(len(feature_names)), 
                   feature_names,
                   rotation=90)
        
    plt.legend(handles=[mpatches.Patch(color=colors(i), label=name) 
                        for i, name in enumerate(class_names)])
    plt.tight_layout()
    if save:
        plt.savefig("plot.png", 
                    dpi=100, 
                    quality=80,
                    optimize=True,
                    progressive=True,
                    transparent=True)
    plt.show()