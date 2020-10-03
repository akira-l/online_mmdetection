import torch
import torch.nn as nn

from ..SSL.model.ssl_model import SSLModel
from ..SSL.model.align_mem import AlignMem  
#from SSL.ot_utils.geometry import 
#from SSL.ot_utils.ot_utils import 
from ..SSL.utils.align_loss import align_loss 
from ..SSL.utils.entropy_2d import Entropy 

from ..SSL.config import Config 


class SSLProc(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.config = Config()
        self.align_model = AlignMem(self.config)


    def get_align_loss(self, otmap_gather_list, pred_gather_list, ctbank_gather_list, err_ctbank_gather_list, use_structure=True, use_context=False, is_reduce=False):
        entopy_val, map_loss, ct_cor_loss, ct_err_loss = align_loss(otmap_gather_list, pred_gather_list, 
                                                                    ctbank_gather_list, 
                                                                    err_ctbank_gather_list, 
                                                                    use_structure,  
                                                                    use_context, 
                                                                    self.config.otmap_struct_max)
        if is_reduce:
            return entropy_val + map_loss + ct_cor_loss + ct_err_loss
        else: 
            return entropy_val, map_loss, ct_cor_loss, ct_err_loss


    def bank_update(self):
        self.align_model.update_bank()




