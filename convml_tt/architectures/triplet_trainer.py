from torch.utils.data import Dataset, DataLoader

from fastai.vision.data import ImageItemList
from fastai.vision.image import Image, pil2tensor
from fastai.data_block import get_files, EmptyLabelList
from fastai.basics import *
from fastai.vision.data import ImageDataBunch, channel_view, normalize_funcs
from fastai.vision import Image
from fastai.basic_data import DeviceDataLoader
from fastai.vision import open_image

import numpy as np

from PIL import Image as PILImage

def loss_func(ys, margin=1.00, l2=0.01):
    z_p, z_n, z_d = ys

    l_n = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
    l_d = - torch.sqrt(((z_p - z_d) ** 2).sum(dim=1))
    l_nd = l_n + l_d
    loss = F.relu(l_n + l_d + margin)
    l_n = torch.mean(l_n)
    l_d = torch.mean(l_d)
    l_nd = torch.mean(l_n + l_d)
    loss = torch.mean(loss)
    if l2 != 0:
        loss += l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
    return loss

class TileType:
    """
    Simple enum for mapping into triplet array """
    ANCHOR = 0
    NEIGHBOR = 1
    DISTANT = 2

    NAMES = ["anchor", "neighbor", "distant"]


def normalize_triplet(x:TensorImage, mean:FloatTensor,std:FloatTensor)->TensorImage:
    "Normalize `x` with `mean` and `std`."
    return [
        (x_-mean[...,None,None]) / std[...,None,None]
        for x_ in x
    ]

def denormalize_triplet(x:List[TensorImage], mean:FloatTensor,std:FloatTensor, do_x:bool=True)->TensorImage:
    "Denormalize `x` with `mean` and `std`."
    return [
        x_.cpu().float()*std[...,None,None] + mean[...,None,None] 
        if do_x else x_.cpu()
        for x_ in x
    ]

def _normalize_triplet_batch(b:Tuple[Tensor,Tensor], mean:FloatTensor, std:FloatTensor, do_x:bool=True, do_y:bool=False)->Tuple[Tensor,Tensor]:
    "`b` = `x`,`y` - normalize `x` array of imgs and `do_y` optionally `y`."
    x,y = b
    mean, std = mean.to(y.device), std.to(y.device)
    if do_x:
        x = normalize_triplet(x, mean, std)
    if do_y:
        raise Exception("`y` isn't used for the triplet trainer, we shouldn't"
                        " be trying to normalize it")
    return x,y

def normalize_triplet_funcs(mean:FloatTensor, std:FloatTensor, do_x:bool=True, do_y:bool=False)->Tuple[Callable,Callable]:
    "Create normalize/denormalize func using `mean` and `std`, can specify `do_y` and `device`."
    mean,std = tensor(mean),tensor(std)
    return (partial(_normalize_triplet_batch, mean=mean, std=std, do_x=do_x, do_y=do_y),
            partial(denormalize_triplet, mean=mean, std=std, do_x=do_x))


class MultiImageDataBunch(ImageDataBunch):

    def normalize(self, stats:Collection[Tensor]=None, do_x:bool=True, do_y:bool=False)->None:
        "Add normalize transform using `stats` (defaults to `DataBunch.batch_stats`)"
        if getattr(self,'norm',False): raise Exception('Can not call normalize twice')
        if stats is None: self.stats = self.batch_stats()
        else:             self.stats = stats
        self.norm,self.denorm = normalize_triplet_funcs(*self.stats, do_x=do_x, do_y=do_y)
        self.add_tfm(self.norm)
        return self

    def batch_stats(self, funcs:Collection[Callable]=None)->Tensor:
        "Grab a batch of data and call reduction function `func` per channel"
        funcs = ifnone(funcs, [torch.mean,torch.std])
        #x = self.one_batch(ds_type=DatasetType.Valid, denorm=False)[0].cpu()
        
        # one_batch gives (x,y) pair on first dim, next dim is going to be the number of images
        # xs = [b.cpu() for b in self.one_batch(ds_type=DatasetType.Valid, denorm=False)[0]]
        # return [[func(channel_view(x), 1) for func in funcs] for x in xs]
        
        x = self.one_batch(ds_type=DatasetType.Valid, denorm=False)[0][0].cpu()
        return [func(channel_view(x), 1) for func in funcs]

    def show_batch(self, rows:int=5, ds_type:DatasetType=DatasetType.Train, reverse:bool=False, **kwargs)->None:
        raise NotImplementedError("Leif: haven't made this work with the triplet trainer yet")

class UnlabelledTripletsList(EmptyLabelList):
    pass




class NPMultiImageList(ImageItemList):
    c = 100

    TILE_FILENAME_FORMAT = "{triplet_id:05d}_{tile_type}.png"
    TRIPLET_META_FILENAME_FORMAT = "{triplet_id:05d}_meta.yaml"

    _bunch = MultiImageDataBunch
    class ImagesList(list):
        def __init__(self, src_path, id, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.id = id
            self.src_path = src_path

        @property
        def size(self):
            return self[0].size

        def apply_tfms(self, tfms, **kwargs):
            items = []
            for item in self:
                items.append(item.apply_tfms(tfms, **kwargs))
            return items

    def open(self, fns, div:bool=True):
        src_path = fns[0].parent
        triplet_id = int(fns[0].name.split('_')[0])

        images = self.ImagesList(id=triplet_id, src_path=src_path)

        for fn_ in fns:
            x = open_image(fn_)
            images.append(x)
        return images

    @classmethod
    def from_folder(cls, path:PathOrStr='.', **kwargs)->ItemList:
        path = Path(path)

        files_by_type = []
        for tt in TileType.NAMES:
            fn_tiletype = cls.TILE_FILENAME_FORMAT.format(tile_type=tt,
                                                          triplet_id=0)
            fn_tiletype = fn_tiletype.replace('00000', '*')
            files_by_type.append(list(path.glob(fn_tiletype)))

        num_files_by_type = [len(files) for files in files_by_type]
        if len(set(num_files_by_type)) != 1:
            raise Exception("There appear to an uneven number of anchor,"
                            " neighbor or distanta tiles.")

        files = zip(*files_by_type)

        return cls(files, path=path, processor=None, **kwargs)

    def label_empty(self, **kwargs):
        "Label every item with an `UnlabelledTripletsList`."
        kwargs['label_cls'] = UnlabelledTripletsList
        return self.label_from_func(func=lambda o: 0., **kwargs)


def loss_batch(model:nn.Module, xb:Tensor, yb:Tensor, loss_func:OptLossFunc=None, opt:OptOptimizer=None,
               cb_handler:Optional[CallbackHandler]=None)->Tuple[Union[Tensor,int,float,str]]:
    "Calculate loss and metrics for a batch, call out to callbacks as necessary."
    cb_handler = ifnone(cb_handler, CallbackHandler())
    if not is_listy(xb): xb = [xb]
    if not is_listy(yb): yb = [yb]
    out = [model(x) for x in xb]
    out = cb_handler.on_loss_begin(out)
        
    #out = cb_handler.on_loss_begin(out)

    if not loss_func: return to_detach(out), yb[0].detach()
    
    #print(out)
    loss = loss_func(out)

    if opt is not None:
        loss = cb_handler.on_backward_begin(loss)
        loss.backward()
        cb_handler.on_backward_end()
        opt.step()
        cb_handler.on_step_end()
        opt.zero_grad()

    return loss.detach().cpu()


def monkey_patch_fastai():
    try:
        print(loss_batch_orig)
    except:
        import fastai.basic_train
        loss_batch_orig = fastai.basic_train.loss_batch
    fastai.basic_train.loss_batch = loss_batch

# for backward compatability
NPMultiImageItemList = NPMultiImageList
