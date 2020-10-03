import os 

class Config():
    def __init__(self, ):
        super().__init__()
        self.para()

    def para(self):
        self.numcls = 80
        self.bank_pick_num = 5
        self.otmap_struct_max = 20
        self.bank_dim = 512  

        self.otmap_thresh = 0



