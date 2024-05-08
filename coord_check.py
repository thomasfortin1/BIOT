from mup.coord_check import plot_coord_data, _record_coords
from nhanes_unsupervised_pretrain_test import prepare_dataloader, Args
from model import UnsupervisedPretrain
import torch
import mup
from mup import set_base_shapes
import torch.nn.functional as F
import pandas as pd

def get_coord_data(models, dataloader, optcls, nsteps=3,
                dict_in_out=False, flatten_input=False, flatten_output=False, 
                output_name='loss', lossfn='xent', filter_module_by_name=None,
                fix_data=True, cuda=True, nseeds=1, 
                output_fdict=None, input_fdict=None, param_fdict=None,
                show_progress=True, one_hot_target=False):
    '''Custom get_coord_data() for biot unsupervised pre-training
    taken from Mup and modified by Thomas

    Train the models in `models` with optimizer given by `optcls` and data from
    `dataloader` for `nsteps` steps, and record coordinate statistics specified
    by `output_fdict`, `input_fdict`, `param_fdict`. By default, only `l1` is
    computed for output activations of each module.

    Inputs:
        models: 
            a dict of lazy models, where the keys are numbers indicating width.
            Each entry of `models` is a function that instantiates a model given
            nothing.
        dataloader:
            an iterator whose elements are either Huggingface style dicts, if
            `dict_in_out` is True, or (input, label). If `fix_data` is True
            (which is the default), then only the first element of `dataloader`
            is used in a loop and the rest of `dataloder` is ignored.
        optcls: 
            a function so that `optcls(model)` gives an optimizer used to train
            the model.
        nsteps: 
            number of steps to train the model
        dict_in_out:
            whether the data loader contains Huggingface-style dict input and
            output. Default: False
        flatten_input:
            if not `dict_in_out`, reshape the input to be
            `input.view(input.shape[0], -1)`. Typically used for testing MLPs.
        flatten_output:
            if not `dict_in_out`, reshape the label to be `label.view(-1,
            input.shape[-1])`.
        output_name:
            if `dict_in_out`, this is the key for the loss value if the output
            is a dict. If the output is not a dict, then we assume the first
            element of the output is the loss.
        lossfn:
            loss function to use if not `dict_in_out`. Can be either a string from
            [`xent`, 'mse', 'nll', 'l1'] or a python `callable` such that
            `lossfn(output, target)` returns the loss value. Examples of valid
            `callable`s are `F.cross_entropy`, `F.mse_loss`, etc, where `F` is
            `torch.nn.functional`. Default: 'xent'
        filter_module_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be recorded.
        cuda:
            whether to use cuda or not. Default: True
        nseeds:
            number of times to repeat the training, each with different seeds.
        output_fdict, input_fdict, param_fdict: 
            function dicts to be used in `_record_coords`. By default, only `l1`
            is computed for output activations of each module.
        show_progress:
            show progress using tqdm. Default: True
        one_hot_target:
            convert target label into a one-hot vector. This typically is only
            used for `'mse'` or `'l1'` losses in classification tasks.
            Default: False
    Output:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'module', 't'` as well as names of statistics recorded, such
        as `'l1'` (see `FDICT` for other premade statistics that can be
        collected).
        
    Breaking Changes:
        In v1.0.0, when `lossfn=='mse'`, the target is automatically converted
        to a one hot vector before loss computation. Starting in v1.1.0, this
        behavior is turned off, and the user needs to explicitly turn on this
        behavior by setting `one_hot_target=True`.
    
    '''
    T=.2
    df = []
    if fix_data:
        batch = next(iter(dataloader))
        dataloader = [batch] * nsteps
    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=nseeds * len(models))
    for i in range(nseeds):
        torch.manual_seed(i)
        for width, model in models.items():
            model = model()
            model = model.train()
            if cuda:
                model.cuda()
                # device = torch.device('cuda')
                # model.to(device)
            optimizer = optcls(model)
            for batch_idx, batch in enumerate(dataloader, 1):
                remove_hooks = []
                # add hooks
                for name, module in model.named_modules():
                    if filter_module_by_name and not filter_module_by_name(name):
                        continue
                    remove_hooks.append(module.register_forward_hook(
                        _record_coords(df, width, name, batch_idx,
                            output_fdict=output_fdict,
                            input_fdict=input_fdict,
                            param_fdict=param_fdict)))

                samples = batch.cuda()
                masked_emb, samples_emb = model(samples, 0)
                
                samples_emb = F.normalize(samples_emb, dim=1, p=2)
                masked_emb = F.normalize(masked_emb, dim=1, p=2)
                N = samples_emb.shape[0]

                # representation similarity matrix, NxN
                logits = torch.mm(samples_emb, masked_emb.t()) / T
                labels = torch.arange(N).to(logits.device)
                loss = F.cross_entropy(logits, labels, reduction="mean")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # remove hooks
                for handle in remove_hooks:
                    handle.remove()

                if batch_idx == nsteps: break
            if show_progress:
                pbar.update(1)
    if show_progress:
        pbar.close()
    return pd.DataFrame(df)


# construct a dictionary of lazy μP models with differing widths
def lazy_model(size):
    # `set_base_shapes` returns the model
    return lambda: set_base_shapes(UnsupervisedPretrain(emb_size=size, heads=size//8, depth=4, n_channels=3, encoder_var=1, use_mup=True), '/home/workplace/thomas/BIOT/base_shapes_0.bsh')
    # Note: any custom initialization with `mup.init` would need to
    # be done inside the lambda as well

embs = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 768, 1024, 1536, 2048]

models = {emb: lazy_model(emb) for emb in embs}

# models = {
#     32: lazy_model(32),
#     40: lazy_model(40),
#     48: lazy_model(48),
#     56: lazy_model(56),
#     64: lazy_model(64),
#     128: lazy_model(128),
#     256: lazy_model(256),
#     512: lazy_model(512),
#     1024: lazy_model(1024),
#     2048: lazy_model(2048)
#     }
# make a dataloader with small batch size/seq len
#   just for testing
args = Args()
args.root = "/media/data_ssd/data/nhanes"
args.batch_size = 256
args.lr = .001
dataloader = prepare_dataloader(args)
# record data from the model activations over a few steps of training
# this returns a pandas dataframe

def get_trainable(model):
    params = model.parameters()
    return params

# Correct implementation with MuAdamW
optcls = lambda model: mup.optim.MuAdamW(get_trainable(model), lr=args.lr)

# Incorrect implementation with Adam
# optcls = lambda model: torch.optim.Adam(get_trainable(model), lr=args.lr)


df = get_coord_data(models, dataloader, optcls, fix_data=True)
# This saves the coord check plots to filename.
file_name = 'coord_check_save'
plot_coord_data(df, save_to=file_name, legend=False)
# folder = 'coord_checks/'
# print(df['t'].unique())
# for module in iter(set(df['module'])):
#     plot_coord_data(df[df['module'] == module], save_to = folder + module + '.png')
# If you are in jupyter notebook, you can also do
#   `plt.show()`
# to show the plot
