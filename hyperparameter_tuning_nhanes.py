import optuna
from optuna.integration import PyTorchLightningPruningCallback
# from model import UnsupervisedPretrain
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import mup
from mup import set_base_shapes
from nhanes_unsupervised_pretrain_test import prepare_dataloader, LitModel_supervised_pretrain
from run_binary_supervised import LitModel_finetune, prepare_TUAB_dataloader
from model import BIOTClassifier
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


EPOCHS = 1
# lr is a hyperparameter for tuning
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256
NUM_WORKERS = 32
KAGGLE_ROOT = "/home/workplace/thomas/BIOT/unlabeled_preprocessed" 
LONG_TERM_ROOT = "/home/workplace/thomas/BIOT/unlabeled_long_term_movement"


class Args:
    epochs = EPOCHS
    lr = .00428
    encoder_var = 1
    weight_decay = WEIGHT_DECAY
    batch_size = BATCH_SIZE
    num_workers = NUM_WORKERS
    root = "/media/data_ssd/data/nhanes"
    emb_size = 256
    depth = 4
    heads = emb_size//8
    cutoff = None
    hop_length = 40
    n_fft = 80


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    # n_fft = trial.suggest_int("n_fft", 40, 160, step=8)
    # hop_length = n_fft//2
    # encoder_var = trial.suggest_float("enc_var", 1e-3, 1e+3)


    # cutoff = trial.suggest_float("cutoff", 1, 50, log=True)
    
    args = Args()
    args.lr = lr
    args.weight_decay = weight_decay
    # args.n_fft = n_fft
    # args.hop_length = hop_length
    # args.encoder_var = encoder_var

    train_loader = prepare_dataloader(args)
    args.root = "/media/data_ssd/data/nhanes_val"
    val_loader = prepare_dataloader(args, shuffle=False)
    
    model = LitModel_supervised_pretrain(args, f"log-pretrain/unsupervised_hyp_tuning/checkpoints", use_mup=True)
    set_base_shapes(model, "/home/workplace/thomas/BIOT/base_shapes_LitModel.bsh")
    print('set the base shapes')
    
    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        benchmark=True,
        enable_checkpointing=False,
        max_steps=100000,
        fast_dev_run=False,
        profiler="simple"
    )

    # hyperparameters = dict(lr=lr, encoder_var=encoder_var)
    # trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, val_loader)

    return trainer.callback_metrics["val_loss"].item()



class Fine_tune_args:
    epochs = 100
    lr = None
    encoder_var = 1
    weight_decay = None
    batch_size = 256
    num_workers = 32
    emb_size = 512
    depth = 4
    heads = emb_size//8
    cutoff = None
    hop_length = 128
    n_fft = 256
    pretrain_model_path = "/media/data_ssd/results/eeg/multi_gpu_test_with_correct_normalization_checkpoints/multi_gpu_test_with_correct_normalization_512-unsupervised/checkpoints/epoch=24_step=400000.ckpt"
    n_classes = 1
    in_channels = 16
    sampling_rate = 256
    sample_length = 10
    token_size = 256
    scale = None
    dataset = "TUAB"

def objective_fine_tune(lr, weight_decay):
    # lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    print("started a trial with lr: ", lr, " weight_decay: ", weight_decay)

    args = Fine_tune_args()
    args.lr = lr
    args.weight_decay = weight_decay

    train_loader, test_loader, val_loader = prepare_TUAB_dataloader(args)

    use_mup=True

    model = BIOTClassifier(
        n_classes=args.n_classes,
        # set the n_channels according to the pretrained model if necessary
        n_channels=args.in_channels,
        n_fft=args.token_size,
        hop_length=args.hop_length,
        scaling=(1/args.scale) if args.scale else None,
        use_mup=use_mup,
        emb_size=512,
    )


    # load from a lightning checkpoint
    checkpoint = torch.load(args.pretrain_model_path)
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        if 'model.biot.' in k:
            state_dict[k.replace('model.biot.', '')] = v
    model.biot.load_state_dict(state_dict=state_dict)
    print(f"load pretrain model from {args.pretrain_model_path}")
    set_base_shapes(model, 'BIOT/base_shapes_BIOTClassifier_2classes.bsh', rescale_params=False)

    lightning_model = LitModel_finetune(args, model, use_mup)

    # logger and callbacks
    version = f"hypreparameter_tuning_for_TUAB"

    early_stop_callback = EarlyStopping(
        monitor="val_auroc", patience=5, verbose=False, mode="max"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auroc',
        mode='max',
        dirpath=f'./checkpoints/{version}',
        filename='hypereparameter_tuning_{epoch:02d}_{step}-val_auroc{val_auroc:.6f}',
        auto_insert_metric_name=False
     )

    trainer = pl.Trainer(
        devices=[0,1],
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        # auto_select_gpus=True,
        benchmark=True,
        enable_checkpointing=True,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        val_check_interval=0.25,
    )

    # train the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # get the best val auroc
    val_auroc = checkpoint_callback.best_model_score

    del train_loader
    del val_loader

    del lightning_model
    torch.cuda.empty_cache()

    return val_auroc


if __name__ == '__main__':
    # torch.set_float32_matmul_precision('medium')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10, n_jobs=1)

    # lrs = [0.002, 0.0005]
    # wds = [1e-6, 1e-5, 1e-4]

    # with open('grid_search_results.txt', 'a') as f:
    #             f.write(f"starting grid search\n")

    # grid = {}
    # for lr in lrs:
    #     for wd in wds:
    #         grid[(lr, wd)] = objective_fine_tune(lr, wd)
    #         # wait for the gpu to be ready
    #         with open('grid_search_results.txt', 'a') as f:
    #             f.write(f"lr: {lr}, wd: {wd}, val_auroc: {grid[(lr, wd)]}\n")
    
    # print(grid)
    # print(max(grid.items(), key=lambda x: x[1]))
