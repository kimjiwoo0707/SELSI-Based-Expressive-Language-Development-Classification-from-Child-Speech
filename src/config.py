import torch

class Args:
    IMAGE_SIZE = (224, 224)
    n_class = 5
    epoch = 25
    learning_rate = 1e-4
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_root = "/content/drive/MyDrive/Dataset_final/train"
    val_root   = "/content/drive/MyDrive/Dataset_final/validation"
    test_root  = "/content/drive/MyDrive/Dataset_final/test"

    best_ckpt_path = "./best/eta_min=1e-9.pth"
    acc_plot_path  = "./best/accuracy.png"
    loss_plot_path = "./best/loss.png"
    cm_dir         = "./confusion_matrix"

    weight_decay = 0.001
    eta_min = 1e-9
    t_max = 25

    seed = 15
