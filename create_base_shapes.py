from mup import make_base_shapes
from model import UnsupervisedPretrain
from unsupervised_pretrain_test import LitModel_supervised_pretrain
from model import BIOTClassifier

# base_model = UnsupervisedPretrain(emb_size=256, heads=8, depth=4, n_channels=3)
# delta_model = UnsupervisedPretrain(emb_size=512, heads=32, depth=4, n_channels=3)
# filename = 'base_shapes_0.bsh'

# make_base_shapes(base_model, delta_model, filename)

class Args:
    emb_size = None
    heads = None
    depth = 4
    encoder_var = 1

base_args = Args()
base_args.emb_size = 256
base_args.heads = 256//8

delta_args = Args()
delta_args.emb_size = 512
delta_args.heads = 512//32

# base_model = LitModel_supervised_pretrain(base_args, "")
# delta_model = LitModel_supervised_pretrain(delta_args, "")
# filename = 'base_shapes_LitModel_test.bsh'


base_model = BIOTClassifier(
    n_classes=2,
    # set the n_channels according to the pretrained model if necessary
    n_channels=16,
    n_fft=256,
    hop_length=128,
    emb_size=256, 
    heads=256//8, 
    depth=4,
    use_mup=True,
)

delta_model = BIOTClassifier(
    n_classes=2,
    n_channels=16,
    n_fft=256,
    hop_length=128,
    emb_size=512, 
    heads=512//32, 
    depth=4,
    use_mup=True,
)

filename = 'base_shapes_BIOTClassifier_2classes.bsh'
make_base_shapes(base_model, delta_model, filename)