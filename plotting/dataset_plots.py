import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def save_thesis_visualization(
    dataset_name,
    samples_per_class=10,
    output_dir="~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures/thesis",
):
    # 1. Setup Pathing
    full_path = os.path.expanduser(output_dir)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Created directory: {full_path}")

    # 2. Configuration and Loading
    if dataset_name == "MNIST":
        # Invert: Background white (1.0), Digits black (0.0)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: 1.0 - x)]
        )
        dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        classes = [str(i) for i in range(10)]
        cmap = "gray"
        show_labels = False  # MNIST image in your reference had no side labels
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        classes = dataset.classes
        cmap = None
        show_labels = True  # CIFAR image in your reference had labels
    else:
        raise ValueError("Dataset not supported.")

    # 3. Organize data
    targets = np.array(dataset.targets)
    fig, axes = plt.subplots(
        len(classes),
        samples_per_class,
        figsize=(samples_per_class * 1.0, len(classes) * 1.0),
    )

    # Minimize spacing between subplots
    plt.subplots_adjust(
        left=0.15 if show_labels else 0.02,
        right=0.98,
        top=0.98,
        bottom=0.02,
        wspace=0.05,
        hspace=0.05,
    )

    for i, class_name in enumerate(classes):
        class_indices = np.where(targets == i)[0]
        selected_indices = np.random.choice(
            class_indices, samples_per_class, replace=False
        )

        # Only add labels for CIFAR-10 to match your reference images
        if show_labels:
            axes[i, 0].annotate(
                class_name,
                xy=(-0.3, 0.5),
                xycoords="axes fraction",
                ha="right",
                va="center",
                fontsize=16,
                fontweight="bold",
            )

        for j, idx in enumerate(selected_indices):
            ax = axes[i, j]
            img, _ = dataset[idx]

            img_display = (
                img.squeeze().numpy()
                if dataset_name == "MNIST"
                else img.permute(1, 2, 0).numpy()
            )

            ax.imshow(img_display, cmap=cmap)
            ax.axis("off")

    # 4. Save with tight bounding box to remove white borders
    save_path = os.path.join(full_path, f"{dataset_name.lower()}_grid.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close()
    print(f"Successfully saved to: {save_path}")


# Generate and save both
save_thesis_visualization("CIFAR10", samples_per_class=10)
save_thesis_visualization("MNIST", samples_per_class=15)
