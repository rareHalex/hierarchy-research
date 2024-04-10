from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models
from tqdm import tqdm
from pathlib import Path
import numpy as np
from metrics.hierarchy import Hierarchy, SubtreeInProb, SubtreeIS

path_dir = 'sd-dog/archive (24)/sd/dogg'
class_dir = 'sd-dog/archive (24)/sd/dogg/dog'
model_name = 'resnet18'

model = getattr(torchvision.models, model_name)(weights="IMAGENET1K_V1")
model.eval()
transform = torchvision.models.get_model_weights(model_name).IMAGENET1K_V1.transforms()
batch_size = 4
logits_list_dict = defaultdict(list)

dataset = ImageFolder(path_dir, transform=transform)
classes_list, _ = dataset.find_classes(class_dir)

dataloader = DataLoader(dataset, batch_size=4)
for images_batch, classes_batch in tqdm.tqdm(dataloader):
    logits_batch = model(images_batch).cpu().detach().numpy()

for logits, class_id in zip(logits_batch, classes_batch):
    logits_list_dict[classes_list[class_id]].append(logits)

logits_dict = {'dog.n.01': np.stack(v) for k, v in logits_list_dict.items()}
save_path = Path('/working/')
save_path.mkdir(parents=True, exist_ok=True)
save_path = save_path / f"{'metrics'}_logits.npz"
np.savez(save_path, **logits_dict)

logits_path = {"model_1": "/kaggle/working/ww_logits.npz"}
dataset_classes_path = Path("/kaggle/input/rl-asset/imagenet.txt")
dataset_hierarchy = Hierarchy(dataset_classes_path.resolve())



synset_num_children = {}
in_prob_metric = SubtreeInProb(dataset_hierarchy, 2)
is_metric = SubtreeIS(dataset_hierarchy, 2)
for synset in dataset_hierarchy.get_all_synsets(remove_leaves=True):
    n_ch = len(dataset_hierarchy.get_classifiable_subtree(synset))
    synset_num_children[synset.name()] = n_ch

more_than_one_child_fn = lambda x: synset_num_children[x] > 1

sorted_keys = np.array([k for k, v in sorted(synset_num_children.items(), key=lambda x: x[0])])
n_ch_array = np.array([v for k, v in sorted(synset_num_children.items(), key=lambda x: x[0])])
max_is_unnormed = np.log(n_ch_array[n_ch_array != 1]).mean()
model_results = {k: (in_prob_metric.compute_metric(v), is_metric.compute_metric(v))for k, v in logits_path.items()}

print("Model", "ISP", "SCS", sep="\t")
for k, v in model_results.items():
    avg_isp = v[0]["average"]
    avg_scs = v[1]["average"] / max_is_unnormed
    print(k, f"{avg_isp:.2f}", f"{avg_scs:.2f}", sep="\t")